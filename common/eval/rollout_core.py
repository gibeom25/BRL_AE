"""Environment loop: wait steps -> (replan: generate chunk H, execute first `n_exec`) until success / max_steps."""
from __future__ import annotations

import time

import numpy as np

DUMMY_ACTION = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)  # gripper open, no motion


def run_episode(env, policy, init_state, max_steps: int = 220, n_wait: int = 10, n_exec: int = 8,
                record: bool = False) -> dict:
    """policy.reset(); policy.act(obs) -> (H, 7) unnormalized actions. Success = env.check_success()/done."""
    env.reset()
    obs = env.set_init_state(init_state)
    # wait for objects to settle (LIBERO convention). Uses env.step so obs-level noise (LIBERO-plus Noise) is applied.
    for _ in range(n_wait):
        obs, _, _, _ = env.step(DUMMY_ACTION)
    policy.reset()
    frames, t, success, n_plans, t0 = [], 0, False, 0, time.time()
    while t < max_steps and not success:
        chunk = policy.act(obs)
        n_plans += 1
        for a in chunk[:n_exec]:
            obs, _, done, _ = env.step(a.astype(np.float64))
            t += 1
            if record:
                frames.append(np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], 1)[::-1, ::-1])
            if done or env.check_success():
                success = True
                break
            if t >= max_steps:
                break
    return {"success": bool(success), "steps": t, "n_plans": n_plans, "sec": round(time.time() - t0, 2),
            "frames": frames if record else None}
