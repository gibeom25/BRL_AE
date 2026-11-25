"""Pitfall 4 check: cached h_tokens vs online h (rollout path).

For sampled frames (task, demo, t):
  cache        : outputs/experiment1/cache/h_tokens (batch-64 encode of hdf5 frame)
  online_h5    : VLMFeatureExtractor.encode(hdf5 frame, batch=1)          -> batch-size effect only
  online_env   : encode(env render at recorded state[t+1], batch=1)        -> real rollout path (obs[t] = after action t)
Reference scales: cache[t] vs cache[t+8] (temporal change within one replan), cache vs other task at same t.
Metric: relative L2 error ||a-b||/||b|| and mean token cosine. Output: outputs/experiment1/cache/h_online_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from common.eval import perturbations as PT  # noqa: E402  (first: libjpeg/wand load order)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.data import libero_dataset as LD  # noqa: E402
from common.models.vlm_features import VLMFeatureExtractor  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402


def metr(a, b):
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    return {"rel_l2": float((a - b).norm() / b.norm()),
            "cos_mean": float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()),
            "rel_l2_img": float((a[:12] - b[:12]).norm() / b[:12].norm()),
            "rel_l2_txt": float((a[12:] - b[12:]).norm() / b[12:].norm())}


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(None, overrides=args.override, smoke=args.smoke, seed=args.seed)
    cfg.setdefault("seed", 0)
    cfg.setdefault("cache_dir", "outputs/experiment1/cache")
    cfg.setdefault("tasks", [0, 4, 8])
    cfg.setdefault("ts", [0, 30, 70])
    cfg["out_dir"] = os.path.join(cfg["cache_dir"], "h_online_check_logs")
    os.chdir(REPO)
    setup_run(cfg)
    idx = json.load(open(os.path.join(cfg["cache_dir"], "h_index.json")))
    split = json.load(open(idx["split_path"]))
    ex = idx["extractor"]
    ext = VLMFeatureExtractor(model_name=ex["model_name"], h_layer=ex["h_layer"], img_pool=ex["img_pool"], n_txt_pool=ex["n_txt_pool"])
    rows = []
    for tid in cfg["tasks"]:
        t = split["tasks"][tid]
        ti = idx["tasks"][tid]
        cache = np.load(os.path.join(cfg["cache_dir"], ti["file"]), mmap_mode="r")
        other = np.load(os.path.join(cfg["cache_dir"], idx["tasks"][(tid + 1) % 10]["file"]), mmap_mode="r")
        demo = t["val"][0]
        o, n = ti["demo_offsets"][demo]
        env = PT.make_env("libero_spatial", t["task_name"], 128)
        env.seed(0)
        env.reset()
        with h5py.File(t["file"], "r") as h5:
            g = h5["data"][demo]
            front = g["obs"]["agentview_rgb"][()]
            states = g["states"][()]
        for ts in cfg["ts"]:
            if ts + 9 >= n:
                continue
            c = np.asarray(cache[o + ts])
            h5_online = ext.encode(front[ts:ts + 1], t["instruction"])[0].cpu().numpy()
            obs = env.regenerate_obs_from_state(states[ts + 1])
            env_online = ext.encode(obs["agentview_image"][None], t["instruction"])[0].cpu().numpy()
            r = {"task": tid, "demo": demo, "t": ts,
                 "cache_vs_online_h5": metr(h5_online, c),
                 "cache_vs_online_env": metr(env_online, c),
                 "ref_cache_t_vs_t+8": metr(np.asarray(cache[o + ts + 8]), c),
                 "ref_cache_vs_other_task_same_t": metr(np.asarray(other[ts]), c),
                 "pixel_mad_h5_vs_env": float(np.abs(front[ts].astype(float) - obs["agentview_image"].astype(float)).mean())}
            rows.append(r)
            print(json.dumps({k: (v["rel_l2"] if isinstance(v, dict) else v) for k, v in r.items()}))
        env.close()
    keys = ["cache_vs_online_h5", "cache_vs_online_env", "ref_cache_t_vs_t+8", "ref_cache_vs_other_task_same_t"]
    summ = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in rows[0][k]} for k in keys}
    out = {"n_frames": len(rows), "summary_mean": summ, "rows": rows}
    json.dump(out, open(os.path.join(cfg["cache_dir"], "h_online_check.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
