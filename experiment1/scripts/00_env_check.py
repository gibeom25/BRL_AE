"""Phase 0 environment check.

1) torch / CUDA build / capability (12,0) / matmul
2) MUJOCO_GL=egl, LIBERO-Spatial clean env reset -> random 10 steps -> save front/wrist images
3) LIBERO-plus Camera / Light / Noise variants: open the actual variant task, save images,
   compare with clean (same init state)
4) VLM (SmolVLM2-500M) load, 1 image + instruction forward, hidden-state shapes,
   image/text token index ranges

Result: outputs/experiment1/env_check/{env_check.json, log.txt, config_used.yaml, *.png}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402

RESULT: dict = {}


def section(name):
    def deco(fn):
        def wrapped(*a, **kw):
            print(f"\n######## {name} ########")
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                RESULT[name] = {"ok": True, **(out or {})}
            except Exception as e:  # noqa: BLE001 - record and continue to next check
                traceback.print_exc()
                RESULT[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            RESULT[name]["sec"] = round(time.time() - t0, 2)
            print(f"[{name}] ok={RESULT[name]['ok']} ({RESULT[name]['sec']}s)")

        return wrapped

    return deco


def save_png(path, arr):
    from PIL import Image

    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------- 1. torch
@section("1_torch")
def check_torch(cfg):
    import torch

    n = int(cfg["torch"]["matmul_n"])
    cap = torch.cuda.get_device_capability(0)
    info = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(cap),
        "capability_is_12_0": cap == (12, 0),
        "arch_list": torch.cuda.get_arch_list(),
    }
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    c = a @ b
    ref = (a[:64].double() @ b.double())[:, :64]
    info["fp32_matmul_max_rel_err"] = float(((c[:64, :64] - ref).abs().max() / ref.abs().max()).item())
    cb = (a.bfloat16() @ b.bfloat16()).float()
    info["bf16_matmul_max_rel_err"] = float(((cb[:64, :64] - ref).abs().max() / ref.abs().max()).item())
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        a.bfloat16() @ b.bfloat16()
    torch.cuda.synchronize()
    info["bf16_tflops"] = round(10 * 2 * n**3 / (time.time() - t0) / 1e12, 1)
    print(json.dumps(info, indent=1))
    assert cap == (12, 0), f"capability {cap} != (12,0)"
    assert torch.version.cuda.startswith("12.8"), f"cuda build {torch.version.cuda} != 12.8"
    return info


# ---------------------------------------------------------------- env helpers
def libero_paths():
    from libero.libero import get_libero_path

    return get_libero_path("bddl_files"), get_libero_path("init_states")


def base_task_names(suite):
    bddl_dir, _ = libero_paths()
    import glob

    # clean base tasks = bddl files whose stem is exactly a key of the original 10 tasks
    init_dir = os.path.join(libero_paths()[1], suite)
    names = sorted(
        os.path.basename(p)[: -len(".pruned_init")]
        for p in glob.glob(os.path.join(init_dir, "*.pruned_init"))
    )
    return names


def make_env(suite, task_name, size):
    """task_name: base name or LIBERO-plus variant name (without .bddl)."""
    from libero.libero.envs import OffScreenRenderEnv

    bddl_dir, _ = libero_paths()
    path = os.path.join(bddl_dir, suite, task_name + ".bddl")
    env = OffScreenRenderEnv(bddl_file_name=path, camera_heights=size, camera_widths=size)
    return env, path


def load_init_states(suite, base_name):
    import torch

    _, init_dir = libero_paths()
    p = os.path.join(init_dir, suite, base_name + ".pruned_init")
    return torch.load(p, weights_only=False), p


def obs_images(obs):
    return obs["agentview_image"], obs["robot0_eye_in_hand_image"]


# ---------------------------------------------------------------- 2. clean env
@section("2_libero_env")
def check_env(cfg, out):
    import numpy as np

    e = cfg["env"]
    names = base_task_names(e["suite"])
    print(f"base tasks ({len(names)}):")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    base = names[e["task_index"]]
    env, path = make_env(e["suite"], base, e["image_size"])
    print("bddl:", path, "| instruction:", env.language_instruction)
    env.seed(cfg["seed"])
    env.reset()
    init_states, init_path = load_init_states(e["suite"], base)
    obs = env.set_init_state(init_states[0])
    keys = {k: list(np.asarray(v).shape) for k, v in obs.items()}
    print("obs keys:", json.dumps(keys, indent=1))
    f0, w0 = obs_images(obs)
    save_png(f"{out}/clean_t0_front_raw.png", f0)
    save_png(f"{out}/clean_t0_wrist_raw.png", w0)
    save_png(f"{out}/clean_t0_front_flip180.png", f0[::-1, ::-1].copy())
    save_png(f"{out}/clean_t0_wrist_flip180.png", w0[::-1, ::-1].copy())
    rng = np.random.default_rng(cfg["seed"])
    for t in range(e["random_steps"]):
        a = rng.uniform(-1, 1, size=7).astype(np.float32)
        obs, r, done, info = env.step(a)
    f1, w1 = obs_images(obs)
    save_png(f"{out}/clean_t{e['random_steps']}_front_raw.png", f1)
    save_png(f"{out}/clean_t{e['random_steps']}_wrist_raw.png", w1)
    out_d = {
        "base_task": base,
        "instruction": env.language_instruction,
        "bddl": path,
        "init_states_file": init_path,
        "n_init_states": int(len(init_states)),
        "obs_keys": keys,
        "action_dim": int(env.env.action_dim),
        "steps_run": e["random_steps"],
        "done": bool(done),
        "wrist_changed_after_steps": float(np.abs(w1.astype(float) - w0.astype(float)).mean()),
    }
    env.close()
    return out_d


# ---------------------------------------------------------------- 3. perturbations
def pick_variant(suite, base, category, difficulty=None):
    from libero.libero import benchmark as bm

    cls_path = os.path.join(os.path.dirname(bm.__file__), "task_classification.json")
    cls = json.load(open(cls_path))[suite]
    cands = [c for c in cls if c["category"] == category and c["name"].startswith(base + "_")]
    # exclude variants that also change language / layout etc.
    if category == "Camera Viewpoints":
        cands = [c for c in cands if c["name"].startswith(base + "_view_")]
    if difficulty is not None:
        d = [c for c in cands if c["difficulty_level"] == difficulty]
        cands = d or cands
    return cands[0], len(cands), cls_path


@section("3_perturbations")
def check_perturb(cfg, out):
    import numpy as np

    e = cfg["env"]
    base = base_task_names(e["suite"])[e["task_index"]]
    init_states, _ = load_init_states(e["suite"], base)
    res = {
        "mechanism": {
            "task_list": "libero/libero/benchmark/libero_suite_task_map.py (libero_task_map['libero_spatial'], 2402 variants, clean 10 base tasks NOT included); "
            "category/difficulty: libero/libero/benchmark/task_classification.json",
            "Camera": "task-name variant `<base>_view_{horiz}_{vert}_{scale%}_{rot}_{vert_tilt}_initstate_{k}` (no bddl file). "
            "Parsed in libero/libero/envs/env_wrapper.py ControlEnv.__init__ -> passed to problem class; "
            "applied at model build in envs/problems/libero_tabletop_manipulation.py Libero_Tabletop_Manipulation._setup_camera "
            "(only `agentview` camera pose changes; wrist camera unaffected). init states: Benchmark.get_task_init_states strips `_view_` -> base .pruned_init",
            "Light": "separate bddl file `<base>_light_{n}.bddl` whose problem name maps (TASK_MAPPING/@register_problem) to class "
            "Libero_Tabletop_Manipulation_tabletop_light_sync_modified_{i} in envs/problems/libero_tabletop_manipulation.py, "
            "which swaps scene_xml to assets/scenes/lights/tabletop_light_sync_modified_{i}.xml (scene-level, affects all cameras)",
            "Noise": "task-name variant `<base>_view_0_0_100_0_0_initstate_0_noise_{n}`; runtime image function in env_wrapper.py "
            "ControlEnv.step / ControlEnv.reset applied ONLY to obs['agentview_image']: n 1-10 motion_blur, 11-20 gaussian_blur, "
            "21-30 zoom_blur, 31-40 fog, 41-50 glass_blur (severity = n mod 10). NOTE: set_init_state/regenerate_obs_from_state does not apply noise.",
        },
        "variants": {},
    }
    clean_env, _ = make_env(e["suite"], base, e["image_size"])
    clean_env.seed(cfg["seed"])
    clean_env.reset()
    obs_c = clean_env.set_init_state(init_states[0])
    zero = np.zeros(7, dtype=np.float32)
    obs_c, *_ = clean_env.step(zero)  # one zero-action step so noise path (step) is exercised
    fc, wc = obs_images(obs_c)
    save_png(f"{out}/cmp_clean_front.png", fc)
    save_png(f"{out}/cmp_clean_wrist.png", wc)
    clean_env.close()

    for key, cat, diff in [("camera", "Camera Viewpoints", 3), ("light", "Light Conditions", None), ("noise", "Sensor Noise", None)]:
        suffix_cfg = e.get(f"{key}_suffix", "auto")
        try:
            if suffix_cfg == "auto":
                v, n_cands, cls_path = pick_variant(e["suite"], base, cat, diff)
                name = v["name"]
                meta = {"difficulty_level": v["difficulty_level"], "id": v["id"], "n_candidates_for_task": n_cands}
            else:
                name, meta = base + "_" + suffix_cfg, {}
            env, path = make_env(e["suite"], name, e["image_size"])
            env.seed(cfg["seed"])
            env.reset()
            obs = env.set_init_state(init_states[0])
            obs, *_ = env.step(zero)
            fv, wv = obs_images(obs)
            save_png(f"{out}/cmp_{key}_front.png", fv)
            save_png(f"{out}/cmp_{key}_wrist.png", wv)
            side = np.concatenate([fc, fv], axis=1)
            save_png(f"{out}/cmp_{key}_vs_clean_front.png", side[::-1, ::-1].copy())
            d = {
                "task_name": name,
                "env_bddl_arg": path,
                "problem_class": type(env.env).__name__,
                "noise_level": getattr(env, "noise", None),
                "front_mean_abs_diff_vs_clean": float(np.abs(fv.astype(float) - fc.astype(float)).mean()),
                "wrist_mean_abs_diff_vs_clean": float(np.abs(wv.astype(float) - wc.astype(float)).mean()),
                **meta,
            }
            env.close()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            d = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
        print(key, json.dumps(d, indent=1))
        res["variants"][key] = d
    return res


# ---------------------------------------------------------------- 4. VLM
@section("4_vlm")
def check_vlm(cfg, out):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    v = cfg["vlm"]
    dtype = getattr(torch, v["dtype"])
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(v["model_name"])
    model = AutoModelForImageTextToText.from_pretrained(v["model_name"], dtype=dtype).cuda().eval()
    load_sec = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    img_path = f"{out}/cmp_clean_front.png"
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
        img = Image.fromarray(np.asarray(img)[::-1, ::-1].copy())  # upright for VLM (see image check)
    else:
        img = Image.fromarray(np.zeros((128, 128, 3), np.uint8))
    instruction = RESULT.get("2_libero_env", {}).get(
        "instruction", "pick up the black bowl and place it on the plate"
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=False)
    inputs = proc(text=prompt, images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0].tolist()
    image_token_id = model.config.image_token_id
    img_idx = [i for i, t in enumerate(ids) if t == image_token_id]
    # contiguous runs of image tokens
    runs = []
    for i in img_idx:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    last_img = img_idx[-1] if img_idx else -1
    tok = proc.tokenizer
    # locate instruction tokens after the last image token
    instr_ids = tok(instruction, add_special_tokens=False)["input_ids"]
    instr_start = None
    for s in range(last_img + 1, len(ids) - len(instr_ids) + 1):
        if ids[s : s + len(instr_ids)] == instr_ids:
            instr_start = s
            break
    with torch.no_grad():
        o = model(**inputs, output_hidden_states=True)
    hs = o.hidden_states
    info = {
        "model": v["model_name"],
        "n_params_M": round(n_params / 1e6, 1),
        "load_sec": round(load_sec, 1),
        "prompt": prompt,
        "instruction": instruction,
        "seq_len": len(ids),
        "pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_token_id": image_token_id,
        "n_image_tokens": len(img_idx),
        "image_token_runs": runs,
        "last_image_token_idx": last_img,
        "instruction_token_range": [instr_start, instr_start + len(instr_ids) - 1] if instr_start is not None else None,
        "post_instruction_tokens": [tok.decode([t]) for t in ids[(instr_start or 0) + len(instr_ids) :]] if instr_start is not None else None,
        "n_hidden_states": len(hs),
        "hidden_state_shape": list(hs[-1].shape),
        "hidden_dtype": str(hs[-1].dtype),
        "hidden_has_nan": bool(torch.isnan(hs[-1]).any().item()),
        "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "decoded_tail": tok.decode(ids[max(0, last_img - 2) :]),
    }
    print(json.dumps(info, indent=1, ensure_ascii=False))
    return info


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--only", nargs="*", default=None, help="subset of 1 2 3 4")
    args = p.parse_args()
    cfg = load_config(
        args.config or os.path.join(REPO, "experiment1/configs/env_check.yaml"),
        overrides=args.override,
        smoke=args.smoke,
        seed=args.seed,
    )
    os.chdir(REPO)
    out = setup_run(cfg)
    only = set(args.only or ["1", "2", "3", "4"])
    print("MUJOCO_GL =", os.environ.get("MUJOCO_GL"))
    if "1" in only:
        check_torch(cfg)
    if "2" in only:
        check_env(cfg, out)
    if "3" in only:
        check_perturb(cfg, out)
    if "4" in only:
        check_vlm(cfg, out)
    RESULT["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(out, "env_check.json")
    prev = json.load(open(path)) if os.path.exists(path) else {}
    prev.update(RESULT)
    json.dump(prev, open(path, "w"), indent=1, ensure_ascii=False)
    print("\nSUMMARY:", {k: v.get("ok") for k, v in RESULT.items() if isinstance(v, dict)})
    print("saved", path)


if __name__ == "__main__":
    main()
