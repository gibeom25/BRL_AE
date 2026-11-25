"""Rollout on LIBERO-Spatial (+ LIBERO-plus perturbations).

python experiment1/scripts/06_rollout.py --model M1_h --perturb none --episodes_per_task 10
  --perturb  none | camera | light | noise
  --h_mode   normal | swap (other task's h_bar, fixed per episode) | zero
Replanning: generate a chunk of H=10, execute the first 8 steps, repeat (max_steps per episode).
Result JSON (plan §3.3): outputs/experiment1/rollouts/{model}__{perturb}__{h_mode}.json
  (with --task_ids subset: ...__{h_mode}__tasks{ids}.json; merge in 07_aggregate.py)

Instruction used for h: the DATASET instruction (split.json), not env.language_instruction
(LIBERO-plus bddl says "pick the akita black bowl ..." while demos say "pick up the black bowl ...").
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from common.eval import perturbations as PT  # noqa: E402  (must be first: libjpeg/wand load order)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.data import libero_dataset as LD  # noqa: E402
from common.eval.rollout_core import run_episode  # noqa: E402
from common.models.flow_matching import euler_sample  # noqa: E402
from common.utils.run_utils import load_config, setup_run  # noqa: E402
from experiment1.models.factory import build_model, uses  # noqa: E402


def latest_ckpt(d):
    cks = glob.glob(os.path.join(d, "step*.pt"))
    return max(cks, key=lambda p: int(re.findall(r"step(\d+)\.pt", p)[0])) if cks else None


class AEPolicy:
    """obs dict -> (H, 7) unnormalized action chunk."""

    def __init__(self, ck: dict, model_id: str, extractor, instruction: str, h_mode: str, h_swap: np.ndarray | None,
                 device: str = "cuda", seed: int = 0):
        M = ck["model_cfg"]
        self.mid, self.M, self.stats = model_id, M, ck["norm_stats"]
        self.model = build_model(model_id, {**M, "pretrained_wrist": False}).to(device).eval()
        self.model.load_state_dict(ck["state_dict"])
        self.ext, self.instruction, self.h_mode, self.dev = extractor, instruction, h_mode, device
        self.need = uses(model_id)
        self.h_swap = None if h_swap is None else torch.tensor(h_swap, device=device).unsqueeze(0)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)
        self.g_hist: list[float] = []

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        b = {}
        if "h" in self.need:
            b["h_tokens"] = self.ext.encode(obs["agentview_image"][None], self.instruction).to(self.dev)
            if self.h_mode == "swap":
                b["h_swap"] = self.h_swap
        if "O" in self.need:
            b["wrist"] = torch.from_numpy(LD.to_model_image(obs["robot0_eye_in_hand_image"])[None]).to(self.dev)
        prop = LD.normalize(LD.proprio_from_env_obs(obs), self.stats["proprio"])
        b["proprio"] = torch.from_numpy(prop[None]).to(self.dev)
        eps = torch.randn((1, self.M["H"], self.M["action_dim"]), generator=self.gen, device=self.dev)
        h_mode = {"normal": "normal", "swap": "swap", "zero": "zero"}[self.h_mode]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a = euler_sample(self.model.make_v_fn(b, train=False, h_mode=h_mode), eps, self.M["n_euler"])
        if self.mid == "M4_moe" and self.model.last_g is not None:
            self.g_hist.append(float(self.model.last_g.float().mean()))
        a = LD.unnormalize(a[0].float().cpu().numpy(), self.stats["actions"])
        a[:, 6] = np.sign(a[:, 6])  # gripper: demos use exactly {-1, +1}
        a[:, 6][a[:, 6] == 0] = -1
        return a


class VerifyPolicy:
    """M5_verify: M1_h generator -> N candidates -> verifier D(O, A) picks best.  TODO(P5/P6): verifier."""

    def __init__(self, *a, **kw):
        raise NotImplementedError("M5_verify rollout: implement after 04_train_verifier.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=os.path.join(REPO, "experiment1/configs/rollout.yaml"))
    p.add_argument("--model", type=str, required=True, choices=["M1_h", "M2_O", "M3_hO", "M4_moe", "M5_verify"])
    p.add_argument("--perturb", type=str, default="none", choices=["none", "camera", "light", "noise"])
    p.add_argument("--h_mode", type=str, default="normal", choices=["normal", "swap", "zero"])
    p.add_argument("--episodes_per_task", type=int, default=None)
    p.add_argument("--task_ids", type=int, nargs="*", default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--smoke", action="store_true", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--override", "-o", nargs="*", default=[])
    args = p.parse_args()
    cfg = load_config(args.config, overrides=args.override, smoke=args.smoke, seed=args.seed)
    os.chdir(REPO)
    if args.episodes_per_task is not None:
        cfg["episodes_per_task"] = args.episodes_per_task
    smoke = bool(cfg.get("smoke"))
    task_ids = args.task_ids if args.task_ids is not None else (cfg["smoke_task_ids"] if smoke else list(range(10)))
    tag = f"{args.model}__{args.perturb}__{args.h_mode}"
    if sorted(task_ids) != list(range(10)):
        tag += "__tasks" + "-".join(map(str, task_ids))
    out_root = os.path.join(cfg["rollout_dir"], "smoke") if smoke else cfg["rollout_dir"]
    cfg.update({"model": args.model, "perturb": args.perturb, "h_mode": args.h_mode, "task_ids": task_ids,
                "out_dir": os.path.join(out_root, "logs", tag)})
    setup_run(cfg)

    split = json.load(open(cfg["split_path"]))
    tasks = {t["task_id"]: t for t in split["tasks"]}
    bases = PT.base_tasks(cfg["suite"])
    assert [tasks[i]["task_name"] for i in range(10)] == bases, "task order mismatch dataset vs LIBERO-plus"

    if args.model == "M5_verify":
        VerifyPolicy()
    ck_dir = os.path.join(cfg["ckpt_root"], args.model + ("_smoke" if smoke else ""))
    ck_path = args.ckpt or latest_ckpt(ck_dir)
    assert ck_path, f"no checkpoint in {ck_dir}"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    print("checkpoint:", ck_path, "step", ck["step"])

    extractor, h_bar = None, None
    if "h" in uses(args.model):
        from common.models.vlm_features import VLMFeatureExtractor

        ex_cfg = ck.get("extractor") or cfg["vlm"]
        extractor = VLMFeatureExtractor(model_name=ex_cfg["model_name"], h_layer=ex_cfg["h_layer"],
                                        img_pool=ex_cfg["img_pool"], n_txt_pool=ex_cfg["n_txt_pool"])
        print("extractor:", extractor.config())
        if args.h_mode == "swap":
            h_bar = np.load(os.path.join(cfg["cache_dir"], "h_bar.npy"))

    per_task, episodes, g_all = {}, [], []
    t_start = time.time()
    for tid in task_ids:
        base = bases[tid]
        inits = PT.init_states(cfg["suite"], base)
        variants = PT.episode_variants(cfg["suite"], base, args.perturb, cfg["episodes_per_task"], cfg["seed"] + tid)
        succ = 0
        for ep in range(cfg["episodes_per_task"]):
            v = variants[ep]
            swap_tid = None
            if args.h_mode == "swap":  # deterministic per (seed, task, episode): independent of task_ids split
                swap_tid = int((tid + np.random.default_rng([cfg["seed"], tid, ep]).integers(1, 10)) % 10)
            policy = AEPolicy(ck, args.model, extractor, tasks[tid]["instruction"], args.h_mode,
                              None if swap_tid is None else h_bar[swap_tid], seed=cfg["seed"] * 1000 + tid * 100 + ep)
            env = PT.make_env(cfg["suite"], v["name"], cfg["image_size"])
            env.seed(cfg["seed"] + ep)
            rec = args.save_video and ep == 0
            r = run_episode(env, policy, inits[ep % len(inits)], max_steps=cfg["max_steps"], n_wait=cfg["n_wait"],
                            n_exec=cfg["n_exec"], record=rec)
            env.close()
            if rec and r["frames"]:
                import imageio

                vd = os.path.join(out_root, "videos", tag)
                os.makedirs(vd, exist_ok=True)
                imageio.mimsave(os.path.join(vd, f"task{tid}_ep{ep}_{'S' if r['success'] else 'F'}.mp4"), r["frames"], fps=20)
            succ += int(r["success"])
            g_all += policy.g_hist
            episodes.append({"task": tid, "episode": ep, "variant": v["name"], "difficulty": v.get("difficulty_level"),
                             "init_idx": ep % len(inits), "swap_task": swap_tid, "success": r["success"],
                             "steps": r["steps"], "sec": r["sec"]})
            print(f"task {tid} ep {ep} {'SUCCESS' if r['success'] else 'fail'} steps={r['steps']} ({r['sec']}s) variant={v['name'][len(base):] or 'clean'}")
        per_task[str(tid)] = [succ, cfg["episodes_per_task"]]
    n = sum(v[1] for v in per_task.values())
    s = sum(v[0] for v in per_task.values())
    res = {"model": args.model, "perturb": args.perturb, "h_mode": args.h_mode, "n": n, "success": s,
           "success_rate": s / n if n else None, "per_task": per_task, "ckpt": ck_path, "seed": cfg["seed"],
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_min": round((time.time() - t_start) / 60, 2),
           "settings": {k: cfg[k] for k in ["max_steps", "n_wait", "n_exec", "image_size", "suite"]},
           "episodes": episodes}
    if g_all:
        res["router_g_mean"] = float(np.mean(g_all))
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"{tag}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"success {s}/{n} -> {path}")


if __name__ == "__main__":
    main()
