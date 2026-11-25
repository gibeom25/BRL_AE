"""P7: offline dependency eval on the validation split (experiment1_plan.md §3.1).

Conditions (applied per model if relevant):
  normal  | h_zero (h_tokens=0) | h_swap (other task's h_bar, per-sample random other task, seeded) | O_zero (O tokens=0)
Same Euler initial noise eps is shared across conditions (seeded per batch).
dependency(cond) = (MSE(cond) - MSE(normal)) / MSE(normal)   [MSE in normalized action space]

Output: outputs/experiment1/offline/{model}.json
--smoke with -o data.fake_h=True works on random h (and random h_bar) before caching.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.data.libero_dataset import ChunkDataset  # noqa: E402
from common.models.flow_matching import euler_sample  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402
from experiment1.models.factory import build_model, uses  # noqa: E402

CONDITIONS = {
    "M1_h": ["normal", "h_zero", "h_swap"],
    "M2_O": ["normal", "O_zero"],
    "M3_hO": ["normal", "h_zero", "h_swap", "O_zero"],
    "M4_moe": ["normal", "h_zero", "h_swap", "O_zero"],
}


def latest_ckpt(d):
    cks = glob.glob(os.path.join(d, "step*.pt"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(re.findall(r"step(\d+)\.pt", p)[0]))


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    args = p.parse_args()
    cfg = load_config(args.config or os.path.join(REPO, "experiment1/configs/offline_eval.yaml"),
                      overrides=args.override, smoke=args.smoke, seed=args.seed)
    os.chdir(REPO)
    mid = args.model
    smoke = bool(cfg.get("smoke"))
    cfg["out_dir"] = os.path.join(cfg["offline_dir"], "smoke") if smoke else cfg["offline_dir"]
    cfg["model_id"] = mid
    out = setup_run(cfg)
    dev = "cuda"
    D = cfg["data"]

    ck_path = args.ckpt or latest_ckpt(os.path.join(cfg["ckpt_root"], mid + ("_smoke" if smoke else "")))
    assert ck_path, f"no checkpoint for {mid}"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    M, stats = ck["model_cfg"], ck["norm_stats"]
    model = build_model(mid, {**M, "pretrained_wrist": False}).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {ck_path} (step {ck['step']})")

    split = json.load(open(D["split_path"]))
    need = uses(mid)
    fake_h = D.get("fake_h", False)
    ds = ChunkDataset(split, stats, "val", H=M["H"], h_cache_dir=D["h_cache_dir"], fake_h=fake_h, n_h=M["n_h"],
                      d_vlm=M["d_vlm"], max_demos_per_task=D.get("max_demos_per_task"),
                      obs_action_shift=ck["data_cfg"].get("obs_action_shift", 0), need_wrist="O" in need,
                      need_h=True, seed=cfg["seed"])
    if fake_h:
        h_bar = np.random.default_rng(cfg["seed"] + 1).standard_normal((len(split["tasks"]), M["n_h"], M["d_vlm"])).astype(np.float32)
    else:
        h_bar = np.load(os.path.join(D["h_cache_dir"], "h_bar.npy"))
    h_bar_t = torch.tensor(h_bar, device=dev)
    n_tasks = len(h_bar)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    conds = CONDITIONS[mid]
    se = {c: 0.0 for c in conds}
    se_task = {c: np.zeros(n_tasks) for c in conds}
    n_task = np.zeros(n_tasks)
    n_samples = 0
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    g_mean = {c: [] for c in conds}
    t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(loader):
            if cfg.get("max_batches") and bi >= cfg["max_batches"]:
                break
            b = {k: v.to(dev) for k, v in b.items()}
            B = b["actions"].shape[0]
            eps = torch.randn(b["actions"].shape, generator=gen, device=dev)  # shared across conditions
            tid = b["task_id"].cpu().numpy()
            other = (tid + rng.integers(1, n_tasks, size=B)) % n_tasks  # always a different task
            b["h_swap"] = h_bar_t[torch.as_tensor(other, device=dev)]
            for c in conds:
                h_mode = {"h_zero": "zero", "h_swap": "swap"}.get(c, "normal")
                O_mode = "zero" if c == "O_zero" else "normal"
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = euler_sample(model.make_v_fn(b, train=False, h_mode=h_mode, O_mode=O_mode), eps, M["n_euler"])
                err = ((pred.float() - b["actions"].float()) ** 2).mean(dim=(1, 2)).cpu().numpy()
                se[c] += err.sum()
                np.add.at(se_task[c], tid, err)
                if mid == "M4_moe" and model.last_g is not None:
                    g_mean[c].append(float(model.last_g.float().mean()))
            np.add.at(n_task, tid, 1)
            n_samples += B
    mse = {c: se[c] / n_samples for c in conds}
    res = {
        "model": mid, "ckpt": ck_path, "ckpt_step": ck["step"], "split": "val", "n": int(n_samples),
        "mse_space": "normalized [-1,1]", "n_euler": M["n_euler"], "shared_eps": True, "seed": cfg["seed"],
        "fake_h": bool(fake_h), "mse": mse,
        "dependency": {c: (mse[c] - mse["normal"]) / mse["normal"] for c in conds if c != "normal"},
        "per_task_mse": {c: (se_task[c] / np.maximum(n_task, 1)).tolist() for c in conds},
        "per_task_n": n_task.astype(int).tolist(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_sec": round(time.time() - t0, 1),
    }
    if mid == "M4_moe":
        res["router_g_mean_last_euler_step"] = {c: float(np.mean(v)) for c, v in g_mean.items() if v}
    path = os.path.join(out, f"{mid}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["model", "n", "mse", "dependency"]}, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
