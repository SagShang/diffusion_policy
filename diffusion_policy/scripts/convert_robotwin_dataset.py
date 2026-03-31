if __name__ == "__main__":
    import pathlib
    import sys

    ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
    sys.path.append(str(ROOT_DIR))

import argparse
import io
import pathlib
import re
import shutil

import h5py
import numpy as np
import zarr
from PIL import Image

from diffusion_policy.common.replay_buffer import ReplayBuffer


def episode_idx(path: pathlib.Path) -> int:
    match = re.search(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Bad episode filename: {path.name}")
    return int(match.group(1))


def decode_image(blob) -> np.ndarray:
    payload = bytes(blob).rstrip(b"\x00")
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return np.moveaxis(array, -1, 0)


def load_episode(path: pathlib.Path, camera_keys: list[str]) -> dict:
    with h5py.File(path, "r") as root:
        qpos = root["joint_action/vector"][:].astype(np.float32)
        episode = {
            "state": qpos[:-1],
            "action": qpos[1:],
        }
        for camera_key in camera_keys:
            images = root[f"observation/{camera_key}/rgb"]
            episode[camera_key] = np.stack(
                [decode_image(x) for x in images[:-1]],
                axis=0,
            )
    return episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/home/wentao/RoboTwin/data/lift_pot")
    parser.add_argument("--output", default="data/datasets/lift_pot")
    parser.add_argument("--camera", nargs="+", default=["head_camera"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input).expanduser().resolve()
    output_dir = pathlib.Path(args.output).expanduser()
    data_dir = input_dir / "data"
    episode_paths = sorted(data_dir.glob("episode*.hdf5"), key=episode_idx)

    if not episode_paths:
        raise ValueError(f"No episodes found under {data_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists, rerun with --overwrite")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.DirectoryStore(str(output_dir))
    )

    total_steps = 0
    for episode_path in episode_paths:
        episode = load_episode(episode_path, args.camera)
        replay_buffer.add_episode(episode, compressors="disk")
        total_steps += len(episode["action"])
        print(f"{episode_path.name}: {len(episode['action'])} steps")

    print(f"saved {len(episode_paths)} episodes / {total_steps} steps to {output_dir}")


if __name__ == "__main__":
    main()
