import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_DINOV3_DEFAULT_WEIGHTS = {
    "dinov3_vits16": "data/pretrained/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "data/pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
}


def _get_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _get_dinov3_repo_path() -> Path:
    return _get_repo_root() / "third_party" / "dinov3"


def _import_dinov3_backbones():
    module_name = "dinov3.hub.backbones"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        repo_path = _get_dinov3_repo_path()
        repo_path_str = str(repo_path)
        if repo_path.exists() and repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)
        return importlib.import_module(module_name)


def _resolve_weights_path(name: str, weights):
    if weights is None:
        default_relpath = _DINOV3_DEFAULT_WEIGHTS.get(name)
        if default_relpath is None:
            return None
        default_path = _get_repo_root() / default_relpath
        return default_path if default_path.exists() else None

    weights_str = str(weights)
    if weights_str.startswith(("http://", "https://", "file://")):
        return weights_str

    weights_path = Path(weights).expanduser()
    if not weights_path.is_absolute():
        weights_path = _get_repo_root() / weights_path
    weights_path = weights_path.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_path}")
    return weights_path


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


class SpatialSoftmaxPool2d(nn.Module):
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = feature_map.shape
        weights = F.softmax(feature_map.flatten(2), dim=-1)

        pos_y = torch.linspace(-1.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        pos_x = torch.linspace(-1.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")
        grid_x = grid_x.reshape(1, 1, -1)
        grid_y = grid_y.reshape(1, 1, -1)

        expected_x = torch.sum(weights * grid_x, dim=-1)
        expected_y = torch.sum(weights * grid_y, dim=-1)
        return torch.cat([expected_x, expected_y], dim=-1)


class DINOv3Encoder(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights=None,
        pretrained=True,
        output_dim=512,
        adapter_dim=256,
        resize_long_edge=320,
        fuse_layers=4,
        pooling="spatial_softmax",
    ):
        super().__init__()

        dinov3_backbones = _import_dinov3_backbones()
        builder = getattr(dinov3_backbones, name)

        resolved_weights = _resolve_weights_path(name=name, weights=weights)
        if resolved_weights is None:
            self.backbone = builder(pretrained=pretrained)
        else:
            self.backbone = builder(pretrained=True, weights=str(resolved_weights))

        self.backbone.requires_grad_(False)
        self.backbone.eval()

        patch_size = int(getattr(self.backbone, "patch_size", 16))
        embed_dim = int(getattr(self.backbone, "embed_dim"))
        total_blocks = len(self.backbone.blocks)
        num_fused_layers = max(1, min(fuse_layers, total_blocks))

        self.preprocess = DINOv3ImagePreprocessor(
            resize_long_edge=resize_long_edge,
            patch_size=patch_size,
        )
        self.layer_indices = list(range(total_blocks - num_fused_layers, total_blocks))

        if num_fused_layers > 1:
            self.layer_weights = nn.Parameter(torch.zeros(num_fused_layers))
        else:
            self.register_parameter("layer_weights", None)

        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim, adapter_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_get_num_groups(adapter_dim), num_channels=adapter_dim),
            nn.GELU(),
        )

        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            pooled_dim = adapter_dim
        elif pooling == "spatial_softmax":
            self.pool = SpatialSoftmaxPool2d()
            pooled_dim = adapter_dim * 2
        else:
            raise ValueError(f"Unsupported DINOv3 pooling: {pooling}")

        self.pooling = pooling
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, output_dim),
        )

    def _fuse_feature_maps(self, feature_maps):
        if len(feature_maps) == 1:
            return feature_maps[0]

        weights = torch.softmax(self.layer_weights, dim=0)
        fused = 0
        for weight, feature_map in zip(weights, feature_maps):
            fused = fused + weight * feature_map
        return fused

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images)

        with torch.no_grad():
            feature_maps = self.backbone.get_intermediate_layers(
                images,
                n=self.layer_indices,
                reshape=True,
                norm=True,
            )

        feature_map = self._fuse_feature_maps(feature_maps)
        feature_map = self.adapter(feature_map)

        if self.pooling == "avg":
            pooled = self.pool(feature_map).flatten(start_dim=1)
        else:
            pooled = self.pool(feature_map)

        return self.head(pooled)
