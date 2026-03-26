import math
from typing import Sequence

import torch
import torch.nn as nn


def normalize_module_paths(module_paths) -> tuple[str, ...]:
    if module_paths is None:
        return ()
    if isinstance(module_paths, str):
        return tuple(path.strip() for path in module_paths.split(",") if path.strip())
    return tuple(str(path) for path in module_paths)


def get_submodule(root_module: nn.Module, module_path: str) -> nn.Module:
    module = root_module
    for part in module_path.split("."):
        module = getattr(module, part)
    return module


def set_submodule(root_module: nn.Module, module_path: str, new_module: nn.Module) -> None:
    parts = module_path.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


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


def inject_lora_into_last_blocks(
    blocks: Sequence[nn.Module],
    *,
    lora_blocks: int,
    lora_targets,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    module_prefix: str = "blocks",
) -> tuple[list[int], list[str]]:
    target_paths = normalize_module_paths(lora_targets)
    if lora_rank <= 0 or lora_blocks <= 0 or len(target_paths) == 0:
        return [], []

    total_blocks = len(blocks)
    num_lora_blocks = min(int(lora_blocks), total_blocks)
    block_indices = list(range(total_blocks - num_lora_blocks, total_blocks))
    lora_module_names = []

    for block_idx in block_indices:
        block = blocks[block_idx]
        for target_path in target_paths:
            module = get_submodule(block, target_path)
            if isinstance(module, LoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"LoRA target '{target_path}' in block {block_idx} must be nn.Linear, got {type(module)}"
                )
            lora_module = LoRALinear(
                base_layer=module,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            set_submodule(block, target_path, lora_module)
            lora_module_names.append(f"{module_prefix}.{block_idx}.{target_path}")

    return block_indices, lora_module_names
