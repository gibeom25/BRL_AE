"""Flow matching (CLAUDE.md §2).

A^tau = tau * A + (1 - tau) * eps,  target velocity = A - eps,  loss = ||(A - eps) - v(A^tau, tau, cond)||^2
Inference: Euler, `n_steps` (default 10), tau: 0 -> 1, starting from eps ~ N(0, I).
"""
from __future__ import annotations

import torch


def sample_tau(b: int, device) -> torch.Tensor:
    return torch.rand(b, device=device)


def fm_noisy(actions: torch.Tensor, tau: torch.Tensor, eps: torch.Tensor):
    t = tau.view(-1, 1, 1)
    a_tau = t * actions + (1 - t) * eps
    target = actions - eps
    return a_tau, target


def fm_loss(v_fn, actions: torch.Tensor, eps: torch.Tensor | None = None, tau: torch.Tensor | None = None):
    """v_fn(a_tau, tau) -> v. Returns (loss, aux dict)."""
    b = actions.shape[0]
    eps = torch.randn_like(actions) if eps is None else eps
    tau = sample_tau(b, actions.device) if tau is None else tau
    a_tau, target = fm_noisy(actions, tau, eps)
    v = v_fn(a_tau, tau)
    loss = ((v.float() - target.float()) ** 2).mean()
    return loss, {"tau": tau}


@torch.no_grad()
def euler_sample(v_fn, eps: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    """eps: (B, H, A) initial noise (pass a seeded tensor to share noise across conditions)."""
    a = eps.clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        tau = torch.full((a.shape[0],), i * dt, device=a.device)
        a = a + dt * v_fn(a, tau).float()
    return a
