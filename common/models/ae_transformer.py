"""Prefix-conditioned Action Expert (AE) backbone.

tokens = [prefix tokens (cond)] + [H noisy-action tokens (+ tau embedding added to each)]
prefix: h_tokens -> Linear (16) | wrist ResNet18 -> Linear (1) | proprio -> MLP (1), each with a type embedding.
Bidirectional transformer encoder; outputs velocity v at the action-token positions.

Ablation knobs (offline eval / rollout): h_mode in {normal, zero, swap}, O_mode in {normal, zero}
- h zero: h_tokens input set to 0 (plan 3.1)
- h swap: caller passes replacement h_tokens (e.g. other task's h_bar)
- O zero: wrist & proprio prefix TOKENS set to 0
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.encoders import ProprioEncoder, TauEmbed, WristEncoder


class CondEncoder(nn.Module):
    """Builds prefix tokens for a given set of conditions ('h', 'O')."""

    def __init__(self, conds: list[str], d: int, d_vlm: int, n_h: int, proprio_dim: int, pretrained_wrist: bool = True):
        super().__init__()
        self.conds = conds
        if "h" in conds:
            self.h_proj = nn.Linear(d_vlm, d)
            self.h_pos = nn.Parameter(torch.randn(1, n_h, d) * 0.02)
        if "O" in conds:
            self.wrist = WristEncoder(d, pretrained=pretrained_wrist)
            self.proprio = ProprioEncoder(proprio_dim, d)
            self.o_type = nn.Parameter(torch.randn(1, 2, d) * 0.02)

    def forward(self, batch: dict, train: bool, h_mode: str = "normal", O_mode: str = "normal", aug=None):
        toks = []
        if "h" in self.conds:
            h = batch["h_tokens"].float()
            if h_mode == "zero":
                h = torch.zeros_like(h)
            elif h_mode == "swap":
                h = batch["h_swap"].float()
            toks.append(self.h_proj(h) + self.h_pos)
        if "O" in self.conds:
            o = torch.cat([self.wrist(batch["wrist"], train=train, aug=aug), self.proprio(batch["proprio"].float())], 1)
            o = o + self.o_type
            if O_mode == "zero":
                o = torch.zeros_like(o)
            toks.append(o)
        return torch.cat(toks, dim=1)

    def o_feature(self, prefix: torch.Tensor) -> torch.Tensor:
        """mean of the O tokens (last 2) — used as f(O) by the MoE router."""
        return prefix[:, -2:].mean(1)


class ActionExpert(nn.Module):
    def __init__(self, conds: list[str], H: int = 10, action_dim: int = 7, d: int = 384, n_layers: int = 6,
                 n_heads: int = 6, d_vlm: int = 960, n_h: int = 16, proprio_dim: int = 8, dropout: float = 0.0,
                 pretrained_wrist: bool = True):
        super().__init__()
        self.conds, self.H, self.action_dim = conds, H, action_dim
        self.cond = CondEncoder(conds, d, d_vlm, n_h, proprio_dim, pretrained_wrist)
        self.act_in = nn.Linear(action_dim, d)
        self.act_pos = nn.Parameter(torch.randn(1, H, d) * 0.02)
        self.tau = TauEmbed(d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, action_dim))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def velocity(self, a_tau: torch.Tensor, tau: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        x = self.act_in(a_tau) + self.act_pos + self.tau(tau).unsqueeze(1)
        z = self.blocks(torch.cat([prefix, x], dim=1))
        return self.out(z[:, -self.H:])

    def make_v_fn(self, batch: dict, train: bool, h_mode="normal", O_mode="normal", aug=None):
        prefix = self.cond(batch, train=train, h_mode=h_mode, O_mode=O_mode, aug=aug)
        return lambda a_tau, tau: self.velocity(a_tau, tau, prefix)
