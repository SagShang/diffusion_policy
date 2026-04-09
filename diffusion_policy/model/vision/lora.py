import math
from typing import Sequence

import torch
import torch.nn as nn


def normalize_module_paths(module_paths: Sequence[str] | str | None) -> tuple[str, ...]:
    if module_paths is None:
        return ()
    if isinstance(module_paths, str):
        return tuple(path.strip() for path in module_paths.split(",") if path.strip())
    return tuple(str(path) for path in module_paths)


def _is_norm_module(module: nn.Module) -> bool:
    if isinstance(
        module,
        (
            nn.LayerNorm,
            nn.GroupNorm,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
        ),
    ):
        return True
    return module.__class__.__name__ == "RMSNorm"


def _unfreeze_norm_modules(module: nn.Module) -> None:
    for submodule in module.modules():
        if _is_norm_module(submodule):
            submodule.requires_grad_(True)


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA can only wrap nn.Linear modules, got {type(base_layer)}")

        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False, **factory_kwargs)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def bias_mask(self):
        return getattr(self.base_layer, "bias_mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        lora_output = self.lora_B(self.lora_A(self.dropout(x)))
        return base_output + lora_output * self.scaling

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, alpha={self.alpha}, "
            f"dropout={getattr(self.dropout, 'p', 0.0)}"
        )


def inject_lora(
    backbone: nn.Module,
    *,
    lora_blocks: int,
    lora_targets,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
) -> nn.Module:
    lora_rank = int(lora_rank)
    lora_alpha = float(lora_alpha)
    lora_dropout = float(lora_dropout)
    lora_blocks = int(lora_blocks)
    target_paths = normalize_module_paths(lora_targets)
    if lora_rank <= 0 or lora_blocks <= 0 or len(target_paths) == 0:
        return backbone

    blocks = backbone.blocks
    total_blocks = len(blocks)
    num_lora_blocks = min(lora_blocks, total_blocks)

    for block in blocks[total_blocks - num_lora_blocks:]:
        for target_path in target_paths:
            target_module = block.get_submodule(target_path)
            if isinstance(target_module, LoRALinear):
                continue
            if not isinstance(target_module, nn.Linear):
                raise TypeError(
                    f"LoRA target '{target_path}' must be nn.Linear, got {type(target_module)}"
                )

            parent_path, _, child_name = target_path.rpartition(".")
            parent_module = block.get_submodule(parent_path) if parent_path else block
            setattr(
                parent_module,
                child_name,
                LoRALinear(
                    base_layer=target_module,
                    rank=lora_rank,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                ),
            )
        _unfreeze_norm_modules(block)

    return backbone
