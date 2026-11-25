"""Merge partial rollout JSONs ({model}__{perturb}__{h_mode}__tasks{ids}.json) into the full-run file.

python experiment1/scripts/06b_merge_rollouts.py --tag M1_h__none__normal [--rollout_dir outputs/experiment1/rollouts]
Refuses to merge if any of tasks 0..9 is missing or duplicated (no partial result is written as complete).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--rollout_dir", default="outputs/experiment1/rollouts")
    a = p.parse_args()
    parts = sorted(glob.glob(os.path.join(a.rollout_dir, f"{a.tag}__tasks*.json")))
    assert parts, f"no parts for {a.tag}"
    per_task, episodes, ckpts, gs, mins = {}, [], set(), [], []
    base = None
    for f in parts:
        r = json.load(open(f))
        base = base or r
        for k, v in r["per_task"].items():
            assert k not in per_task, f"task {k} duplicated ({f})"
            per_task[k] = v
        episodes += r["episodes"]
        ckpts.add(r["ckpt"])
        if "router_g_mean" in r:
            gs.append(r["router_g_mean"])
        mins.append(r["elapsed_min"])
    missing = sorted(set(map(str, range(10))) - set(per_task))
    assert not missing, f"missing tasks {missing} — not merging"
    assert len(ckpts) == 1, f"parts used different checkpoints: {ckpts}"
    n = sum(v[1] for v in per_task.values())
    s = sum(v[0] for v in per_task.values())
    out = {k: base[k] for k in ["model", "perturb", "h_mode"]}
    out.update({"n": n, "success": s, "success_rate": s / n, "per_task": dict(sorted(per_task.items(), key=lambda x: int(x[0]))),
                "ckpt": ckpts.pop(), "seed": base["seed"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_min_max_part": max(mins), "settings": base["settings"], "merged_from": parts,
                "episodes": sorted(episodes, key=lambda e: (e["task"], e["episode"]))})
    if gs:
        out["router_g_mean"] = sum(gs) / len(gs)
    path = os.path.join(a.rollout_dir, f"{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{a.tag}: {s}/{n} = {s / n:.3f} -> {path}")


if __name__ == "__main__":
    main()
