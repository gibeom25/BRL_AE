"""P4: train an Action Expert (M1_h / M2_O / M3_hO / M4_moe) with flow matching.

Outputs (outputs/experiment1/ckpt/{model}/):
  step{N}.pt          model state + model cfg + norm stats + extractor cfg (for rollout)
  train_log.json      loss curve, val MSE (Euler-10, normalized action space, fixed noise)
  g_log.json          (M4 only) router g mean / q10 / q90 every `g_log_every` steps
  log.txt, config_used.yaml
--smoke: few demos, tens of steps. data.fake_h=true uses random h_tokens (before caching).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.data.libero_dataset import ChunkDataset  # noqa: E402
from common.models.flow_matching import euler_sample, fm_loss  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402
from experiment1.models.factory import build_model, uses  # noqa: E402


def to_dev(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(ds, bs, shuffle, workers, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=shuffle, num_workers=workers,
                                       pin_memory=True, persistent_workers=workers > 0 and shuffle, generator=g)


@torch.no_grad()
def val_mse(model, loader, dev, n_steps, seed, max_batches):
    model.eval()
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    se, n = 0.0, 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = to_dev(b, dev)
        eps = torch.randn(b["actions"].shape, generator=gen, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = euler_sample(model.make_v_fn(b, train=False), eps, n_steps)
        se += ((pred.float() - b["actions"].float()) ** 2).sum().item()
        n += b["actions"].numel()
    model.train()
    return se / max(n, 1), n // (b["actions"].shape[1] * b["actions"].shape[2])


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--model", type=str, default=None)
    args = p.parse_args()
    cfg = load_config(args.config, overrides=args.override, smoke=args.smoke, seed=args.seed, model_id=args.model)
    os.chdir(REPO)
    mid = cfg["model_id"]
    cfg["out_dir"] = os.path.join(cfg["ckpt_root"], mid + ("_smoke" if cfg.get("smoke") else ""))
    out = setup_run(cfg)
    dev = "cuda"
    D, M, T = cfg["data"], cfg["model"], cfg["train"]
    split = json.load(open(D["split_path"]))
    stats = json.load(open(D["stats_path"]))
    need = uses(mid)
    common = dict(H=M["H"], h_cache_dir=D["h_cache_dir"], fake_h=D.get("fake_h", False), n_h=M["n_h"],
                  d_vlm=M["d_vlm"], max_demos_per_task=D.get("max_demos_per_task"),
                  obs_action_shift=D.get("obs_action_shift", 0), need_wrist="O" in need,
                  need_h=("h" in need) or mid == "M4_moe", seed=cfg["seed"])
    t0 = time.time()
    ds_tr = ChunkDataset(split, stats, "train", **common)
    ds_va = ChunkDataset(split, stats, "val", **common)
    print(f"dataset: train {len(ds_tr)} / val {len(ds_va)} samples ({time.time() - t0:.1f}s)")
    dl_tr = make_loader(ds_tr, T["batch_size"], True, T["num_workers"], cfg["seed"])
    dl_va = make_loader(ds_va, T["batch_size"], False, 0, cfg["seed"])

    model = build_model(mid, M).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{mid}: {n_params / 1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    steps, warm = int(T["steps"]), int(T["warmup"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    aug = cfg.get("aug", {})
    log = {"model_id": mid, "n_params": n_params, "loss": [], "val": [], "n_train": len(ds_tr), "n_val": len(ds_va)}
    g_log = []
    step, it = 0, iter(dl_tr)
    model.train()
    tl = time.time()
    while step < steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl_tr)
            b = next(it)
        b = to_dev(b, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v_fn = model.make_v_fn(b, train=True, aug=aug)
            loss, aux = fm_loss(v_fn, b["actions"].float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), T["grad_clip"])
        opt.step()
        sched.step()
        step += 1
        if mid == "M4_moe" and (step % T["g_log_every"] == 0 or step == 1):
            g = model.last_g.float().cpu().numpy()
            g_log.append({"step": step, "mean": float(g.mean()), "q10": float(np.quantile(g, 0.1)),
                          "q90": float(np.quantile(g, 0.9)), "std": float(g.std())})
        if step % T["log_every"] == 0 or step == 1:
            sps = T["log_every"] / (time.time() - tl) if step > 1 else 0.0
            tl = time.time()
            log["loss"].append({"step": step, "loss": float(loss.item()), "grad_norm": float(gn), "lr": sched.get_last_lr()[0]})
            extra = f" g={g_log[-1]['mean']:.3f}" if g_log else ""
            print(f"step {step}/{steps} loss {loss.item():.4f} gn {float(gn):.2f} lr {sched.get_last_lr()[0]:.2e} {sps:.1f} it/s{extra}")
        if step % T["val_every"] == 0 or step == steps:
            mse, n = val_mse(model, dl_va, dev, M["n_euler"], cfg["seed"], T["val_max_batches"])
            log["val"].append({"step": step, "val_mse": mse, "n": n})
            print(f"[val] step {step} action MSE (normalized) {mse:.5f} (n={n})")
        if step % T["ckpt_every"] == 0 or step == steps:
            ck = {"model_id": mid, "step": step, "state_dict": model.state_dict(), "model_cfg": M,
                  "norm_stats": stats, "data_cfg": D, "seed": cfg["seed"]}
            if os.path.exists(os.path.join(D["h_cache_dir"], "h_index.json")):
                ck["extractor"] = json.load(open(os.path.join(D["h_cache_dir"], "h_index.json")))["extractor"]
            torch.save(ck, os.path.join(out, f"step{step}.pt"))
            json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
            if g_log:
                json.dump(g_log, open(os.path.join(out, "g_log.json"), "w"), indent=1)
    log["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    log["elapsed_min"] = round((time.time() - t0) / 60, 2)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=1)
    print("done", json.dumps({k: log[k] for k in ["model_id", "n_params", "peak_mem_GB", "elapsed_min"]}))


if __name__ == "__main__":
    main()
