"""Linear probe: does cached h encode the robot state? (CPU, low priority)

Input features (per frame, from outputs/experiment1/cache/h_tokens):
  concat16 : 16 h_tokens concatenated (16*960 = 15360-d)
  meanpool : mean over the 16 tokens (960-d)
Target: EE position (hdf5 obs/ee_pos, same frame index as the cached h — both are the post-action-j observation).
  Object positions: NOT included — hdf5 has no direct object-position key; only the flattened MuJoCo `states`,
  which needs the simulator model (joint qpos addresses) to decode = simulator state restoration.
Ridge regression (primal: eigendecomposition of X^T X once, all alphas reuse it); features standardized with
train stats, targets centered with train mean. Split = split.json (train demos / val demos).
alpha chosen on the val set (as requested) -> val R^2 is slightly optimistic; the label-shuffle control uses the
same procedure, so the comparison is fair.
Controls: shuffled targets (train rows permuted) [required]; random-init VLM h [optional, GPU] -> not run here.
Output: outputs/experiment1/offline/probe_h.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402


def r2(y, p):
    """per-dim R^2 against the eval-set mean."""
    sse = ((y - p) ** 2).sum(0)
    sst = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - sse / sst


def load(split, idx, cache_dir, which, max_demos):
    X = {"train": [], "val": []}
    Y = {"train": [], "val": []}
    T = {"train": [], "val": []}
    for t in split["tasks"]:
        ti = idx["tasks"][t["task_id"]]
        H = np.load(os.path.join(cache_dir, ti["file"]), mmap_mode="r")
        with h5py.File(t["file"], "r") as h5:
            for part in ["train", "val"]:
                for k in t[part][:max_demos] if max_demos else t[part]:
                    o, n = ti["demo_offsets"][k]
                    ee = h5["data"][k]["obs"]["ee_pos"][()]
                    assert len(ee) == n
                    h = np.asarray(H[o:o + n], dtype=np.float32)
                    X[part].append(h.reshape(n, -1) if which == "concat16" else h.mean(1))
                    Y[part].append(ee.astype(np.float64))
                    T[part].append(np.full(n, t["task_id"]))
    return ({p: np.concatenate(X[p]) for p in X}, {p: np.concatenate(Y[p]) for p in Y},
            {p: np.concatenate(T[p]) for p in T})


def ridge_path(Xtr, Ytr_list, Xva, alphas):
    """Return {name: [pred_val for alpha]} for each target matrix in Ytr_list (same X)."""
    X = torch.from_numpy(Xtr)  # float32; accumulate X^T X / X^T Y in float64 chunks (memory)
    d = X.shape[1]
    XtX = torch.zeros(d, d, dtype=torch.float64)
    XtY = [torch.zeros(d, Y.shape[1], dtype=torch.float64) for Y in Ytr_list]
    for i in range(0, len(X), 4096):
        c = X[i:i + 4096].double()
        XtX += c.T @ c
        for j, Y in enumerate(Ytr_list):
            XtY[j] += c.T @ torch.from_numpy(Y[i:i + 4096]).double()
    s, V = torch.linalg.eigh(XtX)
    s = s.clamp_min(0)
    Xv = torch.from_numpy(Xva).double() @ V
    out = []
    for j, Y in enumerate(Ytr_list):
        B = V.T @ XtY[j]  # d x k
        preds = [(Xv @ (B / (s + a)[:, None])).numpy() for a in alphas]
        out.append(preds)
    return out


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(None, overrides=args.override, smoke=args.smoke, seed=args.seed)
    cfg.setdefault("seed", 0)
    cfg.setdefault("cache_dir", "outputs/experiment1/cache")
    cfg.setdefault("threads", 8)
    cfg.setdefault("alphas", [float(a) for a in np.logspace(-1, 6, 15)])
    cfg.setdefault("features", ["meanpool", "concat16"])
    cfg.setdefault("max_demos", None)
    cfg["out_dir"] = "outputs/experiment1/offline/probe_h_logs" + ("_smoke" if cfg.get("smoke") else "")
    if cfg.get("smoke"):
        cfg["max_demos"], cfg["features"] = 2, ["meanpool"]
    os.chdir(REPO)
    setup_run(cfg)
    torch.set_num_threads(int(cfg["threads"]))
    idx = json.load(open(os.path.join(cfg["cache_dir"], "h_index.json")))
    split = json.load(open(idx["split_path"]))
    rng = np.random.default_rng(cfg["seed"])
    res = {"target": "ee_pos (x,y,z) from hdf5 obs/ee_pos",
           "object_positions": "not included: hdf5 has no direct object-position key (only flattened MuJoCo states, "
                               "decoding needs simulator model = state restoration)",
           "split": idx["split_path"], "alpha_selection": "val set (as requested; slightly optimistic, same for control)",
           "alphas": cfg["alphas"], "features": {},
           "controls_not_run": {"random_init_vlm": "optional; needs GPU re-encoding of all frames, GPU busy with training/rollouts"}}
    for feat in cfg["features"]:
        t0 = time.time()
        X, Y, T = load(split, idx, cfg["cache_dir"], feat, cfg["max_demos"])
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        Xtr = ((X["train"] - mu) / sd).astype(np.float32)
        Xva = ((X["val"] - mu) / sd).astype(np.float32)
        del X
        ym = Y["train"].mean(0)
        Ytr = Y["train"] - ym
        Yshuf = Ytr[rng.permutation(len(Ytr))]
        print(f"[{feat}] X train {Xtr.shape} val {Xva.shape} (load {time.time() - t0:.1f}s)")
        (pr, ps) = ridge_path(Xtr, [Ytr, Yshuf], Xva, cfg["alphas"])
        yv = Y["val"]
        r_real = [r2(yv, q + ym) for q in pr]
        r_shuf = [r2(yv, q + ym) for q in ps]
        bi = int(np.argmax([r.mean() for r in r_real]))
        bs = int(np.argmax([r.mean() for r in r_shuf]))
        # per-task R^2 at best alpha (within-task variance only)
        per_task = {}
        for tid in np.unique(T["val"]):
            m = T["val"] == tid
            per_task[int(tid)] = float(r2(yv[m], pr[bi][m] + ym).mean())
        res["features"][feat] = {
            "dim": int(Xtr.shape[1]), "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
            "best_alpha": cfg["alphas"][bi], "val_r2_mean": float(r_real[bi].mean()),
            "val_r2_xyz": r_real[bi].tolist(),
            "shuffle_control": {"best_alpha": cfg["alphas"][bs], "val_r2_mean": float(r_shuf[bs].mean()),
                                "val_r2_xyz": r_shuf[bs].tolist()},
            "val_r2_mean_by_alpha": [float(r.mean()) for r in r_real],
            "val_r2_within_task_mean": float(np.mean(list(per_task.values()))), "val_r2_per_task": per_task,
            "sec": round(time.time() - t0, 1)}
        f = res["features"][feat]
        print(f"[{feat}] val R2 {f['val_r2_mean']:.4f} xyz {np.round(f['val_r2_xyz'], 4)} alpha {f['best_alpha']:.3g} | "
              f"shuffle {f['shuffle_control']['val_r2_mean']:.4f} | within-task {f['val_r2_within_task_mean']:.4f} ({f['sec']}s)")
    res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = "outputs/experiment1/offline/" + ("smoke/probe_h.json" if cfg.get("smoke") else "probe_h.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
