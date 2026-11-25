"""P3: cache h_tokens for every frame of LIBERO-Spatial demos (front image + instruction -> VLM).

Output (cfg.cache_dir):
  h_tokens/task{ii}.npy   (N_frames_task, 16, d_vlm) float16, frames ordered by demo index then t
  h_index.json            per task: demo -> [offset, length]; extractor config; split reference
  h_bar.npy               (n_tasks, 16, d_vlm) float32, per-task mean over TRAIN frames (for h_swap)
  h_stats.json            shape / NaN / throughput / task cosine-similarity matrix
--smoke: 2 batches of task 0 only, reports throughput + ETA, writes to cache_dir/smoke/.
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

from common.data import libero_dataset as LD  # noqa: E402
from common.models.vlm_features import VLMFeatureExtractor, preprocess_images  # noqa: E402
from common.utils.run_utils import add_common_args, load_config, setup_run  # noqa: E402


class _BatchPre(torch.utils.data.Dataset):
    """Item b = preprocessed pixel tensors of frames[b*bs:(b+1)*bs] (CPU work in workers, no CUDA)."""

    def __init__(self, frames, bs, n_batches, image_processor):
        self.frames, self.bs, self.n, self.ip = frames, bs, n_batches, image_processor

    def __len__(self):
        return self.n

    def __getitem__(self, b):
        return preprocess_images(self.ip, self.frames[b * self.bs:(b + 1) * self.bs])


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args()
    cfg = load_config(args.config or os.path.join(REPO, "experiment1/configs/cache_h.yaml"),
                      overrides=args.override, smoke=args.smoke, seed=args.seed)
    os.chdir(REPO)
    smoke = bool(cfg.get("smoke"))
    cache_dir = os.path.join(cfg["cache_dir"], "smoke") if smoke else cfg["cache_dir"]
    cfg["out_dir"] = cache_dir
    setup_run(cfg)
    os.makedirs(os.path.join(cache_dir, "h_tokens"), exist_ok=True)

    split = json.load(open(cfg["split_path"]))
    ext = VLMFeatureExtractor(model_name=cfg["vlm"]["model_name"], h_layer=cfg["vlm"]["h_layer"],
                              img_pool=cfg["vlm"]["img_pool"], n_txt_pool=cfg["vlm"]["n_txt_pool"],
                              dtype=cfg["vlm"]["dtype"])
    print("extractor:", ext.config())
    bs = int(cfg["batch_size"])
    index = {"extractor": ext.config(), "split_path": cfg["split_path"], "tasks": []}
    h_bar, n_done, t_start = [], 0, time.time()
    tasks = split["tasks"][:1] if smoke else split["tasks"]
    debug_printed = False
    for t in tasks:
        tid = t["task_id"]
        with h5py.File(t["file"], "r") as h5:
            keys = LD.demo_keys(h5)
            lens = [h5["data"][k]["actions"].shape[0] for k in keys]
            N = sum(lens)
            out = np.lib.format.open_memmap(os.path.join(cache_dir, "h_tokens", f"task{tid:02d}.npy"), mode="w+",
                                            dtype=np.float16, shape=(N, ext.n_tokens, ext.d_vlm))
            offs, o = {}, 0
            for k, n in zip(keys, lens):
                offs[k] = [o, n]
                o += n
            frames = np.concatenate([h5["data"][k]["obs"]["agentview_rgb"][()] for k in keys])
        n_batches = (N + bs - 1) // bs
        if smoke:
            n_batches = min(n_batches, int(cfg["smoke_batches"]))
        loader = torch.utils.data.DataLoader(
            _BatchPre(frames, bs, n_batches, ext.proc.image_processor), batch_size=None, shuffle=False,
            num_workers=int(cfg["num_workers"]), prefetch_factor=4 if int(cfg["num_workers"]) > 0 else None,
            pin_memory=True)
        torch.cuda.synchronize()
        tb = time.time()
        for b, pix in enumerate(loader):
            sl = slice(b * bs, min((b + 1) * bs, N))
            if not debug_printed:
                h, dbg = ext.encode_preprocessed(pix, t["instruction"], return_debug=True)
                print("token layout:", json.dumps(dbg, ensure_ascii=False))
                index["token_layout_example"] = dbg
                debug_printed = True
            else:
                h = ext.encode_preprocessed(pix, t["instruction"])
            out[sl] = h.cpu().numpy()
            if b % 20 == 0:
                print(f"  task {tid} batch {b}/{n_batches}")
        torch.cuda.synchronize()
        done = min(n_batches * bs, N)
        n_done += done
        fps = done / (time.time() - tb)
        print(f"task {tid}: {done}/{N} frames, {fps:.1f} frames/s")
        out.flush()
        arr = np.asarray(out[:done])
        train_rows = np.concatenate([np.arange(offs[k][0], offs[k][0] + offs[k][1]) for k in t["train"]])
        train_rows = train_rows[train_rows < done]
        h_bar.append(arr[train_rows].astype(np.float32).mean(0))
        index["tasks"].append({"task_id": tid, "task_name": t["task_name"], "instruction": t["instruction"],
                               "file": f"h_tokens/task{tid:02d}.npy", "n_frames": N, "n_cached": done,
                               "demo_offsets": offs})
        del out
    elapsed = time.time() - t_start
    H = np.stack(h_bar)
    np.save(os.path.join(cache_dir, "h_bar.npy"), H)
    json.dump(index, open(os.path.join(cache_dir, "h_index.json"), "w"), indent=1, ensure_ascii=False)

    # stats
    flat = torch.tensor(H.reshape(len(H), -1))
    cos = torch.nn.functional.cosine_similarity(flat[:, None], flat[None], dim=-1).numpy()
    txt = torch.tensor(H[:, -ext.n_txt_pool:].reshape(len(H), -1))
    cos_txt = torch.nn.functional.cosine_similarity(txt[:, None], txt[None], dim=-1).numpy()
    nan = False
    for ti in index["tasks"]:
        a = np.load(os.path.join(cache_dir, ti["file"]), mmap_mode="r")[: ti["n_cached"]]
        nan |= bool(~np.isfinite(a).all())
    total_frames = 0
    for t in split["tasks"]:
        with h5py.File(t["file"], "r") as h5:
            total_frames += sum(h5["data"][k]["actions"].shape[0] for k in LD.demo_keys(h5))
    stats = {"n_tasks_cached": len(index["tasks"]), "frames_cached": n_done, "elapsed_sec": round(elapsed, 1),
             "frames_per_sec": round(n_done / elapsed, 1), "total_frames_all_tasks": total_frames,
             "eta_full_min": round(total_frames / (n_done / elapsed) / 60, 1),
             "h_shape_per_frame": [ext.n_tokens, ext.d_vlm], "has_nan_or_inf": nan,
             "h_bar_cosine_all_tokens": np.round(cos, 4).tolist(),
             "h_bar_cosine_text_tokens": np.round(cos_txt, 4).tolist(),
             "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    if len(H) > 1:
        off = ~np.eye(len(H), dtype=bool)
        stats["cos_offdiag_mean_all"] = float(cos[off].mean())
        stats["cos_offdiag_mean_text"] = float(cos_txt[off].mean())
    json.dump(stats, open(os.path.join(cache_dir, "h_stats.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if "cosine" not in k}, indent=1))
    print("text-token h_bar cosine:\n", np.round(cos_txt, 3))


if __name__ == "__main__":
    main()
