"""M4_moe: v = g * v_h + (1 - g) * v_O.

AE_h (prefix h) and AE_O (prefix O) are each a smaller ActionExpert (n_layers = 3 by default) so the
total size is close to the single-AE models. Router MLP input = [h_bar_pool, f(O), tau_emb] -> g in (0,1).
  h_bar_pool: mean over the 16 raw h_tokens of the sample (d_vlm)
  f(O):       mean of AE_O's two O prefix tokens (d)
`last_g` keeps the most recent router output for logging (g mean / quantiles every 500 steps).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.ae_transformer import ActionExpert
from common.models.encoders import TauEmbed


class MoEAE(nn.Module):
    def __init__(self, H=10, action_dim=7, d=384, n_layers=3, n_heads=6, d_vlm=960, n_h=16, proprio_dim=8,
                 router_hidden=256, pretrained_wrist=True):
        super().__init__()
        kw = dict(H=H, action_dim=action_dim, d=d, n_layers=n_layers, n_heads=n_heads, d_vlm=d_vlm, n_h=n_h,
                  proprio_dim=proprio_dim, pretrained_wrist=pretrained_wrist)
        self.H, self.action_dim = H, action_dim
        self.ae_h = ActionExpert(["h"], **kw)
        self.ae_O = ActionExpert(["O"], **kw)
        self.router_tau = TauEmbed(d)
        self.router = nn.Sequential(nn.Linear(d_vlm + 2 * d, router_hidden), nn.GELU(), nn.Linear(router_hidden, 1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)  # g = 0.5 at init
        self.last_g: torch.Tensor | None = None

    def make_v_fn(self, batch: dict, train: bool, h_mode="normal", O_mode="normal", aug=None):
        pre_h = self.ae_h.cond(batch, train=train, h_mode=h_mode, aug=aug)
        pre_O = self.ae_O.cond(batch, train=train, O_mode=O_mode, aug=aug)
        h_raw = batch["h_tokens"].float()
        if h_mode == "zero":
            h_raw = torch.zeros_like(h_raw)
        elif h_mode == "swap":
            h_raw = batch["h_swap"].float()
        h_bar = h_raw.mean(1)
        f_o = self.ae_O.cond.o_feature(pre_O)

        def v_fn(a_tau, tau):
            g = torch.sigmoid(self.router(torch.cat([h_bar, f_o, self.router_tau(tau)], -1)))  # (B,1)
            self.last_g = g.detach().squeeze(-1)
            g3 = g.unsqueeze(-1)
            return g3 * self.ae_h.velocity(a_tau, tau, pre_h) + (1 - g3) * self.ae_O.velocity(a_tau, tau, pre_O)

        return v_fn
