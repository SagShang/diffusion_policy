import os
import sys

import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from diffusion_policy.model.vision.dinov3_encoder import DINOv3Encoder
from diffusion_policy.model.vision.lora import LoRALinear, inject_lora


class DummyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)


class DummyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DummyAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)


class DummyBackbone(nn.Module):
    def __init__(self, dim: int = 8, depth: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([DummyBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.cls_norm = nn.LayerNorm(dim)
        self.local_cls_norm = nn.LayerNorm(dim)


def test_lora_unfreezes_norms_in_selected_blocks():
    backbone = DummyBackbone()
    backbone.requires_grad_(False)

    inject_lora(
        backbone,
        lora_blocks=2,
        lora_targets=("attn.qkv", "attn.proj"),
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )

    selected_blocks = backbone.blocks[1:]
    frozen_blocks = backbone.blocks[:1]

    assert not backbone.norm.weight.requires_grad
    assert not backbone.norm.bias.requires_grad
    assert not backbone.cls_norm.weight.requires_grad
    assert not backbone.cls_norm.bias.requires_grad
    assert not backbone.local_cls_norm.weight.requires_grad
    assert not backbone.local_cls_norm.bias.requires_grad

    for block in selected_blocks:
        assert isinstance(block.attn.qkv, LoRALinear)
        assert isinstance(block.attn.proj, LoRALinear)
        assert block.norm1.weight.requires_grad
        assert block.norm1.bias.requires_grad
        assert block.norm2.weight.requires_grad
        assert block.norm2.bias.requires_grad
        assert block.attn.qkv.lora_A.weight.requires_grad
        assert block.attn.qkv.lora_B.weight.requires_grad
        assert not block.attn.qkv.base_layer.weight.requires_grad
        assert not block.attn.proj.base_layer.weight.requires_grad

    for block in frozen_blocks:
        assert isinstance(block.attn.qkv, nn.Linear)
        assert isinstance(block.attn.proj, nn.Linear)
        assert not block.norm1.weight.requires_grad
        assert not block.norm1.bias.requires_grad
        assert not block.norm2.weight.requires_grad
        assert not block.norm2.bias.requires_grad


def test_encoder_trainable_norm_unfreezes_backbone_output_norms():
    encoder = DINOv3Encoder.__new__(DINOv3Encoder)
    nn.Module.__init__(encoder)
    encoder.backbone = DummyBackbone()
    encoder.backbone.requires_grad_(False)

    encoder._unfreeze_norms()

    assert encoder.backbone.norm.weight.requires_grad
    assert encoder.backbone.norm.bias.requires_grad
    assert encoder.backbone.cls_norm.weight.requires_grad
    assert encoder.backbone.cls_norm.bias.requires_grad
    assert encoder.backbone.local_cls_norm.weight.requires_grad
    assert encoder.backbone.local_cls_norm.bias.requires_grad
    assert not encoder.backbone.blocks[0].norm1.weight.requires_grad
    assert not encoder.backbone.blocks[0].norm1.bias.requires_grad
    assert not encoder.backbone.blocks[0].norm2.weight.requires_grad
    assert not encoder.backbone.blocks[0].norm2.bias.requires_grad
