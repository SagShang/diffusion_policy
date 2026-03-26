from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dill
import hydra
import numpy as np
import torch
import yaml

from diffusion_policy.env_runner.dp_runner import DPRunner
from diffusion_policy.workspace.robotworkspace import RobotWorkspace


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct: dict[str, Any]):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


class DiffusionPolicyModel:
    def __init__(self, checkpoint_path: str, config_path: str, device: str):
        runtime_cfg = self._load_runtime_config(config_path)
        self.policy = self._load_policy(checkpoint_path, device=device)
        self.runner = DPRunner(
            n_obs_steps=runtime_cfg["n_obs_steps"],
            n_action_steps=runtime_cfg["n_action_steps"],
        )

    @staticmethod
    def _load_runtime_config(config_path: str) -> dict[str, int]:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return {
            "n_obs_steps": int(cfg["n_obs_steps"]),
            "n_action_steps": int(cfg["n_action_steps"]),
        }

    @staticmethod
    def _load_policy(checkpoint_path: str, device: str):
        with open(checkpoint_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location=device)

        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.to(torch.device(device))
        policy.eval()
        return policy

    def get_action(self, obs: dict[str, Any]):
        return self.runner.get_action(self.policy, obs)

    def update_obs(self, obs: dict[str, Any]):
        self.runner.update_obs(obs)
        return None

    def reset_obs(self):
        self.runner.reset_obs()
        return None


class PolicyServer:
    def __init__(self, model: DiffusionPolicyModel, host: str, port: int):
        self.model = model
        self.host = host
        self.port = port

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = sock.recv(min(remaining, 4096))
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _send_payload(sock: socket.socket, payload: dict[str, Any]):
        raw = numpy_to_json(payload).encode("utf-8")
        sock.sendall(len(raw).to_bytes(4, "big"))
        sock.sendall(raw)

    def _dispatch(self, request: dict[str, Any]):
        cmd = request.get("cmd")
        obs = request.get("obs")
        if cmd not in {"get_action", "update_obs", "reset_obs"}:
            raise ValueError(f"Unsupported command: {cmd}")

        method = getattr(self.model, cmd)
        return method(obs) if obs is not None else method()

    def _handle_client(self, client_socket: socket.socket, addr: tuple[str, int]):
        with client_socket:
            while True:
                len_bytes = self._recv_exact(client_socket, 4)
                if len_bytes is None:
                    print(f"client disconnected: {addr}")
                    return

                request_size = int.from_bytes(len_bytes, "big")
                request_bytes = self._recv_exact(client_socket, request_size)
                if request_bytes is None:
                    print(f"client disconnected during request: {addr}")
                    return

                try:
                    request = json_to_numpy(request_bytes.decode("utf-8"))
                    result = self._dispatch(request)
                    self._send_payload(client_socket, {"ok": True, "result": result})
                except Exception as exc:
                    self._send_payload(
                        client_socket,
                        {
                            "ok": False,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )

    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            print(f"robotwin diffusion policy server listening on {self.host}:{self.port}")

            while True:
                client_socket, addr = server_socket.accept()
                print(f"client connected: {addr}")
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True,
                )
                thread.start()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt file")
    parser.add_argument("--config", required=True, help="Path to runtime config yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    model = DiffusionPolicyModel(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    server = PolicyServer(model=model, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
