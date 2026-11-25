"""LIBERO hdf5 loading, train/val split, normalization stats, shared obs conventions.

Conventions here are the single source of truth for BOTH training data and rollout:
- task order: sorted base task names (== sorted `*_demo.hdf5` stems without `_demo`)
- proprio (8D): ee_pos(3) + ee_ori(3, axis-angle) + gripper_states(2)
  rollout side: robot0_eef_pos, quat2axisangle(robot0_eef_quat), robot0_gripper_qpos
  (equivalence verified by experiment1 data check, see docs/documents/logs/)
- images: `to_model_image` is applied to hdf5 images AND rollout obs images.
"""
from __future__ import annotations

import glob
import json
import os
import random

import numpy as np

PROPRIO_KEYS = ("ee_pos", "ee_ori", "gripper_states")
PROPRIO_DIM = 8
ACTION_DIM = 7

# Set by the data check (P2). True = rotate 180 deg (flip H and W) to make the image upright.
# hdf5 and env obs share the same raw convention (opengl), so the same flag applies to both.
FLIP_180 = True


def to_model_image(img: np.ndarray) -> np.ndarray:
    """(…, H, W, 3) uint8 raw (hdf5 or env obs) -> upright image used by VLM / wrist encoder."""
    if FLIP_180:
        img = img[..., ::-1, ::-1, :]
    return np.ascontiguousarray(img)


def task_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*_demo.hdf5")))
    return files


def task_name_from_file(path: str) -> str:
    return os.path.basename(path)[: -len("_demo.hdf5")]


def demo_keys(h5) -> list[str]:
    return sorted(h5["data"].keys(), key=lambda k: int(k.split("_")[1]))


def proprio_from_h5(demo_grp) -> np.ndarray:
    return np.concatenate([demo_grp["obs"][k][()] for k in PROPRIO_KEYS], axis=-1).astype(np.float32)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite convention (x, y, z, w) -> axis-angle (3,). Same as robosuite.utils.transform_utils."""
    quat = np.array(quat, dtype=np.float64)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def proprio_from_env_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def make_split(data_dir: str, n_val: int = 5, seed: int = 0) -> dict:
    import h5py

    rng = random.Random(seed)
    split = {"seed": seed, "n_val_per_task": n_val, "tasks": []}
    for i, f in enumerate(task_files(data_dir)):
        with h5py.File(f, "r") as h5:
            keys = demo_keys(h5)
            instr = json.loads(h5["data"].attrs["problem_info"])["language_instruction"]
        shuffled = keys[:]
        rng.shuffle(shuffled)
        val = sorted(shuffled[:n_val], key=lambda k: int(k.split("_")[1]))
        train = [k for k in keys if k not in val]
        split["tasks"].append(
            {"task_id": i, "task_name": task_name_from_file(f), "file": f, "instruction": instr,
             "train": train, "val": val}
        )
    return split


def compute_stats(split: dict) -> dict:
    """min/max (+mean/std for reference) of actions and proprio over TRAIN demos only."""
    import h5py

    acts, props = [], []
    for t in split["tasks"]:
        with h5py.File(t["file"], "r") as h5:
            for k in t["train"]:
                g = h5["data"][k]
                acts.append(g["actions"][()].astype(np.float32))
                props.append(proprio_from_h5(g))
    A = np.concatenate(acts)
    P = np.concatenate(props)

    def s(x):
        return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
                "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "n": int(len(x))}

    return {"actions": s(A), "proprio": s(P), "proprio_keys": list(PROPRIO_KEYS)}


def normalize(x: np.ndarray, st: dict) -> np.ndarray:
    lo, hi = np.asarray(st["min"], np.float32), np.asarray(st["max"], np.float32)
    return 2.0 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1.0


def unnormalize(x: np.ndarray, st: dict) -> np.ndarray:
    lo, hi = np.asarray(st["min"], np.float32), np.asarray(st["max"], np.float32)
    return (x + 1.0) / 2.0 * np.maximum(hi - lo, 1e-6) + lo


# ----------------------------------------------------------------------------- training dataset
class ChunkDataset:
    """(h_tokens[t], wrist[t], proprio[t]) -> actions[t+shift : t+shift+H] (normalized, repeat-last padded).

    split_name: "train" | "val". Data per task held in RAM (wrist uint8 in model orientation).
    h source: `{h_cache_dir}/h_tokens/task{ii}.npy` + `h_index.json` (row offsets per demo),
              or random N(0,1) when fake_h=True (smoke tests before caching).
    obs_action_shift: 0 = LIBERO default pairing (obs[j] is the obs AFTER action j, paired with action j);
                      1 = pair obs[j-1] (= state before action j) with action j. See logs/20261125.md.
    """

    def __init__(self, split: dict, stats: dict, split_name: str, H: int, h_cache_dir: str | None = None,
                 fake_h: bool = False, n_h: int = 16, d_vlm: int = 960, max_demos_per_task: int | None = None,
                 obs_action_shift: int = 0, need_wrist: bool = True, need_h: bool = True, seed: int = 0):
        import h5py

        self.H, self.fake_h, self.n_h, self.d_vlm = H, fake_h, n_h, d_vlm
        self.need_wrist, self.need_h = need_wrist, need_h
        self.task_ids, self.wrist, self.proprio, self.actions, self.h = [], [], [], [], []
        self.index = []  # (task_slot, row, end_row_exclusive)
        h_index = None
        if need_h and not fake_h:
            with open(os.path.join(h_cache_dir, "h_index.json")) as f:
                h_index = {t["task_id"]: t for t in json.load(f)["tasks"]}
        rng = np.random.default_rng(seed)
        for slot, t in enumerate(split["tasks"]):
            demos = t[split_name][:max_demos_per_task] if max_demos_per_task else t[split_name]
            W, P, A, Hs = [], [], [], []
            if need_h and not fake_h:
                h_all = np.load(os.path.join(h_cache_dir, h_index[t["task_id"]]["file"]), mmap_mode="r")
                offs = h_index[t["task_id"]]["demo_offsets"]
            row = 0
            with h5py.File(t["file"], "r") as h5:
                for k in demos:
                    g = h5["data"][k]
                    a = g["actions"][()].astype(np.float32)
                    p = proprio_from_h5(g)
                    n = len(a)
                    if need_wrist:
                        W.append(to_model_image(g["obs"]["eye_in_hand_rgb"][()]))
                    if need_h:
                        if fake_h:
                            Hs.append(rng.standard_normal((n, n_h, d_vlm)).astype(np.float16))
                        else:
                            o, ln = offs[k]
                            assert ln == n, (k, ln, n)
                            Hs.append(np.asarray(h_all[o:o + n]))
                    A.append(normalize(a, stats["actions"]))
                    P.append(normalize(p, stats["proprio"]))
                    # obs row r pairs with action rows r+shift ...; valid obs rows: 0 .. n-1-shift
                    for r in range(n - obs_action_shift):
                        self.index.append((slot, row + r, row + n, obs_action_shift))
                    row += n
            self.task_ids.append(t["task_id"])
            self.actions.append(np.concatenate(A))
            self.proprio.append(np.concatenate(P))
            self.wrist.append(np.concatenate(W) if need_wrist else None)
            self.h.append(np.concatenate(Hs) if need_h else None)
        self.index = np.asarray(self.index, dtype=np.int64)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        slot, r, end, shift = self.index[i]
        a0 = r + shift
        rows = np.minimum(np.arange(a0, a0 + self.H), end - 1)
        item = {
            "actions": self.actions[slot][rows],
            "proprio": self.proprio[slot][r],
            "task_id": np.int64(self.task_ids[slot]),
        }
        if self.need_wrist:
            item["wrist"] = self.wrist[slot][r]
        if self.need_h:
            item["h_tokens"] = self.h[slot][r]
        return item
