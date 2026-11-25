"""LIBERO-plus perturbation wrapper — based on code read in env_check (outputs/experiment1/env_check/env_check.json).

How LIBERO-plus applies each perturbation (verified, see logs/20261125.md):
- Camera: task-NAME variant `<base>_view_{h}_{v}_{scale%}_{rot}_{tilt}_initstate_0` (no bddl file on disk).
  ControlEnv.__init__ (libero/libero/envs/env_wrapper.py) parses it -> Libero_Tabletop_Manipulation._setup_camera
  moves ONLY the `agentview` camera. Wrist image unchanged (env_check: wrist diff 0.0).
- Light: separate bddl `<base>_light_{n}.bddl` -> problem class `..._light_sync_modified_{i}` swaps scene_xml.
  Scene-level: changes BOTH front and wrist images.
- Noise: task-name variant `<base>_view_0_0_100_0_0_initstate_0_noise_{n}`; ControlEnv.step/reset blur
  obs['agentview_image'] only (motion/gaussian/zoom/fog/glass by n). set_init_state() does NOT apply it.
- Clean: LIBERO-plus suite lists only variants (2402); clean = base bddl + base `.pruned_init`.
All variants of a base task use the base task's `.pruned_init` init states.
"""
from __future__ import annotations

import glob
import json
import os

# IMPORT ORDER MATTERS: libero envs import `wand` (conda MagickWand -> conda libtiff -> conda libjpeg).
# If pillow/torchvision (VLM stack) load their bundled libjpeg first, libtiff fails with
# "undefined symbol: jpeg12_write_raw_data". Import this module before torch/transformers/PIL.
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402,F401

import numpy as np

PERTURB_CATEGORY = {"camera": "Camera Viewpoints", "light": "Light Conditions", "noise": "Sensor Noise"}


def libero_paths():
    from libero.libero import get_libero_path

    return get_libero_path("bddl_files"), get_libero_path("init_states")


def base_tasks(suite: str = "libero_spatial") -> list[str]:
    """Sorted base task names — same order as dataset task_id (sorted *_demo.hdf5)."""
    _, init_dir = libero_paths()
    return sorted(os.path.basename(p)[: -len(".pruned_init")] for p in glob.glob(os.path.join(init_dir, suite, "*.pruned_init")))


def _classification(suite: str) -> list[dict]:
    from libero.libero import benchmark as bm

    return json.load(open(os.path.join(os.path.dirname(bm.__file__), "task_classification.json")))[suite]


def variants_for(suite: str, base: str, perturb: str) -> list[dict]:
    """All LIBERO-plus variants of `base` for a perturbation (pure variants only, e.g. camera excludes language+view)."""
    if perturb == "none":
        return [{"name": base, "difficulty_level": 0, "id": None}]
    cat = PERTURB_CATEGORY[perturb]
    prefix = {"camera": base + "_view_", "light": base + "_light_", "noise": base + "_view_"}[perturb]
    out = [c for c in _classification(suite) if c["category"] == cat and c["name"].startswith(prefix)]
    if perturb == "noise":
        out = [c for c in out if "_noise_" in c["name"]]
    return out


def episode_variants(suite: str, base: str, perturb: str, n_episodes: int, seed: int) -> list[dict]:
    """One variant per episode: seeded sample without replacement (cycles if fewer variants than episodes)."""
    vs = variants_for(suite, base, perturb)
    assert vs, f"no {perturb} variants for {base}"
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(vs))
    return [vs[order[i % len(vs)]] for i in range(n_episodes)]


def make_env(suite: str, task_name: str, image_size: int = 128):
    from libero.libero.envs import OffScreenRenderEnv

    bddl_dir, _ = libero_paths()
    path = os.path.join(bddl_dir, suite, task_name + ".bddl")
    return OffScreenRenderEnv(bddl_file_name=path, camera_heights=image_size, camera_widths=image_size)


def init_states(suite: str, base: str):
    import torch

    _, init_dir = libero_paths()
    return torch.load(os.path.join(init_dir, suite, base + ".pruned_init"), weights_only=False)
