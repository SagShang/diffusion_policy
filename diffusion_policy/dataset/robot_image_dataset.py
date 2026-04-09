from typing import Dict
import numba
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import (
    get_identity_image_normalizer,
    get_image_range_normalizer,
)


class RobotImageDataset(BaseImageDataset):

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=128,
        max_train_episodes=None,
        image_normalizer="range",
        shape_meta=None,
    ):

        super().__init__()
        assert shape_meta is not None
        obs_shape_meta = shape_meta["obs"]
        self.rgb_obs_keys = [
            key for key, attr in obs_shape_meta.items()
            if attr.get("type", "low_dim") == "rgb"
        ]
        self.low_dim_obs_keys = [
            key for key, attr in obs_shape_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        ]
        self.obs_key_to_buffer_key = {
            key: self._resolve_buffer_key(key)
            for key in self.rgb_obs_keys + self.low_dim_obs_keys
        }
        replay_buffer_keys = list(
            dict.fromkeys([
                *self.obs_key_to_buffer_key.values(),
                "action",
            ])
        )
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=replay_buffer_keys,
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_normalizer = image_normalizer

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    @staticmethod
    def _resolve_buffer_key(obs_key):
        if obs_key.endswith("_cam"):
            return obs_key[:-4] + "_camera"
        if obs_key == "agent_pos":
            return "state"
        return obs_key

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {"action": self.replay_buffer["action"]}
        for obs_key in self.low_dim_obs_keys:
            data[obs_key] = self.replay_buffer[self.obs_key_to_buffer_key[obs_key]]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        if self.image_normalizer == "range":
            image_normalizer = get_image_range_normalizer()
        elif self.image_normalizer == "identity":
            image_normalizer = get_identity_image_normalizer()
        else:
            raise ValueError(f"Unsupported image_normalizer: {self.image_normalizer}")
        for obs_key in self.rgb_obs_keys:
            normalizer[obs_key] = image_normalizer
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255
        left_cam = np.moveaxis(sample["left_camera"], -1, 1) / 255
        right_cam = np.moveaxis(sample["right_camera"], -1, 1) / 255

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                "left_cam": left_cam,  # T, 3, H, W
                "right_cam": right_cam,  # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        obs = dict()
        for obs_key in self.rgb_obs_keys:
            image = samples[self.obs_key_to_buffer_key[obs_key]].to(device, non_blocking=True)
            obs[obs_key] = image.float() / 255.0
        for obs_key in self.low_dim_obs_keys:
            obs[obs_key] = samples[self.obs_key_to_buffer_key[obs_key]].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": obs,
            "action": action,  # B, T, D
        }


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
