"""Which camera does each LIBERO-plus perturbation change? (slide table)

Same base task + same init state (pruned_init[0]) + one identical dummy step (so step-time Noise is applied),
clean vs camera / light / noise. Metric: mean absolute pixel difference (0-255) on raw 128x128 images,
front (agentview) and wrist (robot0_eye_in_hand), averaged over tasks x variants.

Output: outputs/experiment1/env_check/perturb_matrix.{json,png}
  png: rows = clean/camera/light/noise, cols = front/wrist (upright), labeled with that pair's MAD.
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

import numpy as np  # noqa: E402

from common.data.libero_dataset import to_model_image  # noqa: E402
from common.eval.rollout_core import DUMMY_ACTION  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402

PERTS = ["camera", "light", "noise"]


def render(suite, name, init_state, size, seed):
    env = PT.make_env(suite, name, size)
    env.seed(seed)
    env.reset()
    env.set_init_state(init_state)
    obs, *_ = env.step(DUMMY_ACTION.astype(np.float64))
    f, w = obs["agentview_image"].copy(), obs["robot0_eye_in_hand_image"].copy()
    env.close()
    return f, w


def mad(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def label_grid(rows, labels, col_titles, scale=3):
    from PIL import Image, ImageDraw, ImageFont

    size = rows[0][0].shape[0] * scale
    left, top, pad = 150, 34, 6
    W = left + len(col_titles) * (size + pad)
    Hh = top + len(rows) * (size + pad + 22)
    canvas = Image.new("RGB", (W, Hh), "white")
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    for j, t in enumerate(col_titles):
        d.text((left + j * (size + pad) + 8, 6), t, fill="black", font=font)
    for i, (imgs, (rl, sub)) in enumerate(zip(rows, labels)):
        y = top + i * (size + pad + 22)
        d.text((8, y + size // 2 - 10), rl, fill="black", font=font)
        for j, im in enumerate(imgs):
            x = left + j * (size + pad)
            canvas.paste(Image.fromarray(im).resize((size, size), Image.NEAREST), (x, y))
            d.text((x + 4, y + size + 2), sub[j], fill="black", font=small)
    return canvas


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(None, overrides=args.override, smoke=args.smoke, seed=args.seed)
    cfg.setdefault("seed", 0)
    cfg.setdefault("suite", "libero_spatial")
    cfg.setdefault("tasks", [0, 4, 8])
    cfg.setdefault("variants_per_task", 5)
    cfg.setdefault("grid_task", 0)
    cfg.setdefault("image_size", 128)
    cfg["out_dir"] = "outputs/experiment1/env_check/perturb_matrix_logs"
    if cfg.get("smoke"):
        cfg["tasks"], cfg["variants_per_task"] = [0], 1
    os.chdir(REPO)
    setup_run(cfg)
    bases = PT.base_tasks(cfg["suite"])
    S, sd = cfg["image_size"], cfg["seed"]
    rec = {k: {"front": [], "wrist": [], "wrist_identical": [], "variants": []} for k in PERTS}
    grid = {}
    for tid in cfg["tasks"]:
        base = bases[tid]
        init = PT.init_states(cfg["suite"], base)[0]
        fc, wc = render(cfg["suite"], base, init, S, sd)
        if tid == cfg["grid_task"]:
            grid["clean"] = (fc, wc)
        for k in PERTS:
            vs = PT.episode_variants(cfg["suite"], base, k, cfg["variants_per_task"], sd + tid)
            for vi, v in enumerate(vs):
                fv, wv = render(cfg["suite"], v["name"], init, S, sd)
                rec[k]["front"].append(mad(fv, fc))
                rec[k]["wrist"].append(mad(wv, wc))
                rec[k]["wrist_identical"].append(bool(np.array_equal(wv, wc)))
                rec[k]["variants"].append({"task": tid, "name": v["name"][len(base):], "difficulty": v["difficulty_level"],
                                           "front_mad": rec[k]["front"][-1], "wrist_mad": rec[k]["wrist"][-1]})
                if tid == cfg["grid_task"] and vi == 0:
                    grid[k] = (fv, wv, v["name"][len(base):])
                print(f"task {tid} {k:6s} {v['name'][len(base):]:45s} front {rec[k]['front'][-1]:6.2f} wrist {rec[k]['wrist'][-1]:6.2f}")
    table = {k: {"n": len(rec[k]["front"]),
                 "front_mad_mean": float(np.mean(rec[k]["front"])), "front_mad_std": float(np.std(rec[k]["front"])),
                 "wrist_mad_mean": float(np.mean(rec[k]["wrist"])), "wrist_mad_std": float(np.std(rec[k]["wrist"])),
                 "wrist_identical_frac": float(np.mean(rec[k]["wrist_identical"]))} for k in PERTS}
    out_dir = "outputs/experiment1/env_check"
    json.dump({"metric": "mean abs pixel diff (0-255) vs clean, same task & init state, after 1 identical step",
               "tasks": cfg["tasks"], "variants_per_task": cfg["variants_per_task"], "table": table,
               "details": {k: rec[k]["variants"] for k in PERTS}},
              open(os.path.join(out_dir, "perturb_matrix.json"), "w"), indent=1)
    rows, labels = [], []
    fc, wc = grid["clean"]
    rows.append([to_model_image(fc), to_model_image(wc)])
    labels.append(("clean", ["reference", "reference"]))
    for k in PERTS:
        fv, wv, nm = grid[k]
        rows.append([to_model_image(fv), to_model_image(wv)])
        labels.append((k, [f"MAD {mad(fv, fc):.1f}", f"MAD {mad(wv, wc):.1f}"]))
    img = label_grid(rows, labels, ["front (agentview) -> h", "wrist (eye_in_hand) -> O"])
    img.save(os.path.join(out_dir, "perturb_matrix.png"))
    print("\n| perturbation | front MAD (mean±std) | wrist MAD (mean±std) | wrist identical | n |")
    for k in PERTS:
        t = table[k]
        print(f"| {k} | {t['front_mad_mean']:.1f} ± {t['front_mad_std']:.1f} | {t['wrist_mad_mean']:.1f} ± {t['wrist_mad_std']:.1f} | {t['wrist_identical_frac']*100:.0f}% | {t['n']} |")


if __name__ == "__main__":
    main()
