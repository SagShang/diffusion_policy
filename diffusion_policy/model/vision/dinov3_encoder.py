import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.vision.lora import inject_lora

def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_dinov3_backbone(weights, pretrained: bool = True) -> nn.Module:
    if weights is None:
        raise ValueError("DINOv3 weights path is required")

    repo_path = _get_repo_root() / "third_party" / "dinov3"
    repo_path_str = str(repo_path)
    if importlib.util.find_spec("dinov3") is None and repo_path.exists() and repo_path_str not in sys.path:
        sys.path.insert(0, repo_path_str)
    dinov3_backbones = importlib.import_module("dinov3.hub.backbones")

    builder = dinov3_backbones.dinov3_vits16
    backbone = builder(pretrained=False)
    if not pretrained:
        return backbone

    weights_str = str(weights)
    if weights_str.startswith(("http://", "https://", "file://")):
        state_dict = torch.hub.load_state_dict_from_url(weights_str, map_location="cpu")
    else:
        weights_path = Path(weights).expanduser()
        if not weights_path.is_absolute():
            weights_path = _get_repo_root() / weights_path
        weights_path = weights_path.resolve()
        if not weights_path.exists():
            raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

    backbone.load_state_dict(state_dict, strict=True)
    return backbone


def _get_num_groups(num_channels: int) -> int:
    for num_groups in (32, 16, 8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class DINOv3ImagePreprocessor(nn.Module):
    def __init__(
        self,
        resize_long_edge: int,
        patch_size: int,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.resize_long_edge = resize_long_edge
        self.patch_size = patch_size
        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected images in (B, C, H, W), got {tuple(images.shape)}")

        images = images.to(dtype=torch.float32)
        _, _, height, width = images.shape
        scale = self.resize_long_edge / float(max(height, width))
        resized_height = max(1, int(round(height * scale)))
        resized_width = max(1, int(round(width * scale)))

        if resized_height != height or resized_width != width:
            images = F.interpolate(
                images,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

        images = (images - self.mean) / self.std

        pad_height = (self.patch_size - resized_height % self.patch_size) % self.patch_size
        pad_width = (self.patch_size - resized_width % self.patch_size) % self.patch_size
        if pad_height > 0 or pad_width > 0:
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            images = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        return images


class AttentionPool2d(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(tokens), dim=1)
        return torch.sum(tokens * weights, dim=1)


class SpatialSoftmax2d(nn.Module):

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        weights = F.softmax(feature_map.flatten(2), dim=-1).reshape(batch_size, channels, height, width)
        pos_y = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        pos_x = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        expected_x = torch.sum(weights * grid_x.view(1, 1, height, width), dim=(2, 3))
        expected_y = torch.sum(weights * grid_y.view(1, 1, height, width), dim=(2, 3))
        return torch.cat([expected_x, expected_y], dim=-1)


class DINOv3CameraAdapter(nn.Module):

    def __init__(self, input_dim: int, adapter_dim: int, output_dim: int, pooling: str = "spatial_softmax"):
        super().__init__()
        num_groups = _get_num_groups(adapter_dim)
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, adapter_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=adapter_dim),
            nn.GELU(),
            nn.Conv2d(adapter_dim, adapter_dim, kernel_size=3, padding=1, groups=adapter_dim, bias=False),
            nn.Conv2d(adapter_dim, adapter_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=adapter_dim),
            nn.GELU(),
        )

        if pooling == "attention":
            self.pool = AttentionPool2d(adapter_dim)
            pooled_dim = adapter_dim
        elif pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pooled_dim = adapter_dim
        elif pooling == "spatial_softmax":
            self.pool = SpatialSoftmax2d()
            pooled_dim = adapter_dim * 2
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        self.out = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, output_dim),
        )
        self.pooling = pooling

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        reduced = self.proj(feature_map)
        if self.pooling == "attention":
            pooled = self.pool(reduced.flatten(2).transpose(1, 2))
        elif self.pooling == "avg":
            pooled = self.pool(reduced).flatten(start_dim=1)
        else:
            pooled = self.pool(reduced)
        return self.out(pooled)


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        weights=None,
        pretrained=True,
        freeze_backbone=True,
        output_dim=512,
        adapter_dim=256,
        resize_long_edge=320,
        fuse_layers=1,
        pooling="spatial_softmax",
        lora_rank=0,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_blocks=0,
        lora_targets: Sequence[str] | str = ("attn.qkv", "attn.proj"),
        lora_lr_scale=1.0,
    ):
        super().__init__()

        self.backbone = _build_dinov3_backbone(
            weights=weights,
            pretrained=pretrained,
        )

        self.backbone.requires_grad_(not freeze_backbone)
        if freeze_backbone:
            self.backbone.eval()

        self.lora_lr_scale = float(lora_lr_scale)

        patch_size = int(getattr(self.backbone, "patch_size", 16))
        embed_dim = int(getattr(self.backbone, "embed_dim"))
        total_blocks = len(self.backbone.blocks)
        num_fused_layers = max(1, min(fuse_layers, total_blocks))

        self.preprocess = DINOv3ImagePreprocessor(
            resize_long_edge=resize_long_edge,
            patch_size=patch_size,
        )
        self.layer_indices = list(range(total_blocks - num_fused_layers, total_blocks))
        self.backbone = inject_lora(
            self.backbone,
            lora_blocks=lora_blocks,
            lora_targets=lora_targets,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.backbone_requires_grad = any(param.requires_grad for param in self.backbone.parameters())

        if num_fused_layers > 1:
            self.layer_weights = nn.Parameter(torch.zeros(num_fused_layers))
        else:
            self.register_parameter("layer_weights", None)

        self.adapter = DINOv3CameraAdapter(
            input_dim=embed_dim,
            adapter_dim=adapter_dim,
            output_dim=output_dim,
            pooling=pooling,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.backbone_requires_grad:
            self.backbone.eval()
        return self

    def get_optimizer_param_groups(self, base_lr: float):
        base_lr = float(base_lr)
        grouped_params = {"base": [], "lora": []}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_lora_param = name.startswith("backbone.") and ".lora_" in name
            grouped_params["lora" if is_lora_param else "base"].append(param)

        lr_by_group = {
            "base": base_lr,
            "lora": base_lr * self.lora_lr_scale,
        }
        return [
            {
                "params": params,
                "lr": lr_by_group[group_name],
                "name": group_name,
            }
            for group_name, params in grouped_params.items()
            if params
        ]

    def _fuse_feature_maps(self, feature_maps):
        if len(feature_maps) == 1:
            return feature_maps[0]

        weights = torch.softmax(self.layer_weights, dim=0)
        stacked = torch.stack(feature_maps, dim=1)
        return torch.sum(stacked * weights.view(1, -1, 1, 1, 1), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)

        if self.backbone_requires_grad and torch.is_grad_enabled():
            feature_maps = self.backbone.get_intermediate_layers(
                images,
                n=self.layer_indices,
                reshape=True,
                norm=True,
            )
        else:
            with torch.no_grad():
                feature_maps = self.backbone.get_intermediate_layers(
                    images,
                    n=self.layer_indices,
                    reshape=True,
                    norm=True,
                )

        feature_map = self._fuse_feature_maps(feature_maps)
        return self.adapter(feature_map)
