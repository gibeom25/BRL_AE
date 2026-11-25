"""Observation encoders: wrist ResNet18 (-> 1 token), proprio MLP (-> 1 token), wrist augmentation."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def wrist_augment(x: torch.Tensor, train: bool, crop: float = 0.9, jitter: float = 0.1) -> torch.Tensor:
    """x: (B,3,H,W) float in [0,1]. Train: per-sample random crop (crop*side) resized back + mild
    brightness/contrast jitter. Eval: deterministic center crop with the same scale."""
    B = x.shape[0]
    theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = crop
    theta[:, 1, 1] = crop
    if train:
        theta[:, :, 2] = (torch.rand(B, 2, device=x.device, dtype=x.dtype) * 2 - 1) * (1 - crop)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    if train and jitter > 0:
        b = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * jitter
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * c + m) * b
        x = x.clamp(0, 1)
    return x


class WristEncoder(nn.Module):
    """ResNet18 (ImageNet init) -> global pool 512 -> Linear(d)."""

    def __init__(self, d: int, pretrained: bool = True):
        super().__init__()
        import torchvision

        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Identity()
        self.net = net
        self.proj = nn.Linear(512, d)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, wrist_uint8: torch.Tensor, train: bool, aug: dict | None = None) -> torch.Tensor:
        """wrist_uint8: (B,H,W,3) uint8 already in model orientation (to_model_image). Returns (B,1,d)."""
        x = wrist_uint8.permute(0, 3, 1, 2).float() / 255.0
        x = wrist_augment(x, train=train, **(aug or {}))
        x = (x - self.mean) / self.std
        return self.proj(self.net(x)).unsqueeze(1)


class ProprioEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.net(p).unsqueeze(1)


def sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """t in [0,1] (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TauEmbed(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal(tau, self.d))
