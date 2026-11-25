"""P2 data check (LIBERO-Spatial demos).

1) files / demos / keys / frame counts per task
2) action & proprio ranges, gripper action values
3) image orientation + obs/state alignment: render env at recorded sim states, pixel-compare with hdf5 images
   (raw vs flip180, and image[j] vs state[j] / state[j+1]); save side-by-side PNGs
4) proprio mapping: hdf5 (ee_pos, ee_ori, gripper_states) vs env obs -> proprio_from_env_obs
5) split.json (45/5 per task) + train-only normalization stats -> outputs/experiment1/cache/

Results: outputs/experiment1/data_check/data_check.json (+ PNGs), cache/split.json, cache/norm_stats.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from common.data import libero_dataset as LD  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402


def save_png(path, arr):
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(arr)).save(path)


def mad(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def inventory(cfg):
    files = LD.task_files(cfg["data_dir"])
    rows, total = [], 0
    for i, f in enumerate(files):
        with h5py.File(f, "r") as h5:
            keys = LD.demo_keys(h5)
            lens = [h5["data"][k]["actions"].shape[0] for k in keys]
            g = h5["data"][keys[0]]
            obs_keys = {k: list(g["obs"][k].shape[1:]) for k in g["obs"].keys()}
            instr = json.loads(h5["data"].attrs["problem_info"])["language_instruction"]
            conv = h5["data"].attrs.get("macros_image_convention")
        total += sum(lens)
        rows.append({"task_id": i, "task_name": LD.task_name_from_file(f), "instruction": instr,
                     "n_demos": len(keys), "len_min": min(lens), "len_mean": round(float(np.mean(lens)), 1),
                     "len_max": max(lens), "frames": sum(lens), "image_convention": conv})
        print(f"[{i}] demos={len(keys)} frames={sum(lens)} len={min(lens)}-{max(lens)} | {instr}")
    print("obs keys:", obs_keys)
    return {"n_files": len(files), "total_frames": total, "obs_keys": obs_keys, "tasks": rows}


def ranges(cfg, max_demos):
    acts, props = [], []
    for f in LD.task_files(cfg["data_dir"]):
        with h5py.File(f, "r") as h5:
            for k in LD.demo_keys(h5)[:max_demos]:
                acts.append(h5["data"][k]["actions"][()])
                props.append(LD.proprio_from_h5(h5["data"][k]))
    A, P = np.concatenate(acts), np.concatenate(props)
    out = {
        "action_min": A.min(0).round(4).tolist(), "action_max": A.max(0).round(4).tolist(),
        "gripper_action_unique": np.unique(A[:, 6]).round(4).tolist()[:10],
        "proprio_min": P.min(0).round(4).tolist(), "proprio_max": P.max(0).round(4).tolist(),
        "action_has_nan": bool(np.isnan(A).any()), "proprio_has_nan": bool(np.isnan(P).any()),
    }
    print(json.dumps(out, indent=1))
    return out


def env_alignment(cfg, out_dir):
    """Render env at recorded states; compare with hdf5 images (orientation + time offset) and proprio."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    res = []
    files = LD.task_files(cfg["data_dir"])
    for ti in cfg["align_tasks"]:
        f = files[ti]
        name = LD.task_name_from_file(f)
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_spatial", name + ".bddl")
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.seed(0)
        env.reset()
        with h5py.File(f, "r") as h5:
            g = h5["data"][LD.demo_keys(h5)[0]]
            states = g["states"][()]
            front = g["obs"]["agentview_rgb"][()]
            wrist = g["obs"]["eye_in_hand_rgb"][()]
            prop = LD.proprio_from_h5(g)
        T = len(states)
        js = [j for j in cfg["align_steps"] if j + 1 < T] + [T - 2]
        for j in js:
            o_j = env.regenerate_obs_from_state(states[j])
            o_j1 = env.regenerate_obs_from_state(states[j + 1])
            r = {"task_id": ti, "step": j}
            for nm, o in [("state_j", o_j), ("state_j+1", o_j1)]:
                fe, we = o["agentview_image"], o["robot0_eye_in_hand_image"]
                r[f"front_raw_vs_{nm}"] = mad(front[j], fe)
                r[f"front_flip180_vs_{nm}"] = mad(front[j][::-1, ::-1], fe)
                r[f"wrist_raw_vs_{nm}"] = mad(wrist[j], we)
                r[f"proprio_absdiff_vs_{nm}"] = np.abs(prop[j] - LD.proprio_from_env_obs(o)).max().round(5).tolist()
            res.append(r)
            if j == js[0]:
                fe = o_j1["agentview_image"]
                save_png(f"{out_dir}/align_t{ti}_front_h5raw_envraw_h5upright.png",
                         np.concatenate([front[j], fe, LD.to_model_image(front[j])], 1))
                save_png(f"{out_dir}/align_t{ti}_wrist_h5raw_envraw_h5upright.png",
                         np.concatenate([wrist[j], o_j1["robot0_eye_in_hand_image"], LD.to_model_image(wrist[j])], 1))
            print(json.dumps(r))
        env.close()
    return res


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config or os.path.join(REPO, "experiment1/configs/data_check.yaml"),
                      overrides=args.override, smoke=args.smoke, seed=args.seed)
    os.chdir(REPO)
    out = setup_run(cfg)
    R = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("\n### 1 inventory"); R["inventory"] = inventory(cfg)
    print("\n### 2 ranges"); R["ranges"] = ranges(cfg, cfg["range_max_demos"])
    print("\n### 3-4 env alignment"); R["alignment"] = env_alignment(cfg, out)
    print("\n### 5 split + stats")
    split = LD.make_split(cfg["data_dir"], n_val=cfg["n_val"], seed=cfg["seed"])
    stats = LD.compute_stats(split)
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    if not cfg.get("smoke"):
        json.dump(split, open(os.path.join(cfg["cache_dir"], "split.json"), "w"), indent=1)
        json.dump(stats, open(os.path.join(cfg["cache_dir"], "norm_stats.json"), "w"), indent=1)
        print("saved split.json / norm_stats.json to", cfg["cache_dir"])
    R["split_summary"] = {"n_train": sum(len(t["train"]) for t in split["tasks"]),
                          "n_val": sum(len(t["val"]) for t in split["tasks"]),
                          "train_frames_for_stats": stats["actions"]["n"]}
    R["stats_actions"] = {k: np.round(stats["actions"][k], 4).tolist() for k in ["min", "max"]}
    json.dump(R, open(os.path.join(out, "data_check.json"), "w"), indent=1, ensure_ascii=False)
    print("saved", os.path.join(out, "data_check.json"))


if __name__ == "__main__":
    main()
