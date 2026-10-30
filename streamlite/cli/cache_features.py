"""Stage 0: cache g_t and the 1-fps frames (lossless PNG) of every clip (docs/plan.md §4). One GPU, vision tower only.

g_t is computed exactly as the online write path does: frames from `iter_video_frames` (resized to
frame_max_side), `processor.image_processor` on one frame, the frozen vision tower + projector in the
backbone dtype, the merged tokens' pooler_output cast to fp32 and averaged (`LVMMemoryFormer` takes
H.mean(0) of the same tensor). Only the tower's weights are loaded, so the job fits one 11 GB card;
`--verify` compares against the full model's `encode_frame` (streamlite/cli/verify_cache.py does that check).

Sharded by clip (--shard i --num_shards n), resumable: a clip with meta/<uid>.json is skipped.
Decoding and preprocessing run in DataLoader workers; the GPU does one frame per call, as online.
"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from streamlite.cli.common import load_yaml
from streamlite.utils.video import iter_video_frames
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
FRAME_FORMAT = "png"
def clip_list(clean_dir: str, splits) -> list:
    with open(os.path.join(clean_dir, "clips.json")) as f:
        clips = json.load(f)
    order = {s: i for i, s in enumerate(splits)}
    clips = [c for c in clips if c["split"] in order]
    return sorted(clips, key=lambda c: (order[c["split"]], c["clip_uid"]))
class ClipFrames(IterableDataset):
    """Yields ("frame", uid, k, pixel_values, grid_thw, jpeg_bytes, size) per frame and ("done", uid, n, err)."""
    def __init__(self, clips, video_dir, sample_fps, max_side, model_path):
        self.clips, self.video_dir, self.fps, self.max_side, self.model_path = clips, video_dir, sample_fps, max_side, model_path
    def __iter__(self):
        from transformers import AutoProcessor
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        proc = AutoProcessor.from_pretrained(self.model_path).image_processor
        for i, c in enumerate(self.clips):
            if i % nw != wid:
                continue
            uid = c["clip_uid"]
            n_max = int(math.ceil(float(c["duration_sec"]) * self.fps - 1e-6))
            count, err = 0, None
            try:
                for k, (img, _) in enumerate(iter_video_frames(os.path.join(self.video_dir, f"{uid}.mp4"), sample_fps=self.fps, max_side=self.max_side)):
                    if k >= n_max:
                        break
                    p = proc(images=[img], return_tensors="pt")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", compress_level=1)  # lossless: JPEG q95 moved the prompt hidden state more than one real frame step
                    yield ("frame", uid, k, p["pixel_values"], p["image_grid_thw"], buf.getvalue(), img.size)
                    count += 1
            except Exception as e:  # recorded in meta; the clip is then retried on resume
                err = f"{type(e).__name__}: {e}"
            yield ("done", uid, count, err, n_max)
def load_tower(model_path: str, dtype: torch.dtype, device: str):
    """The backbone's vision tower + merger alone, weights `visual.*` from the checkpoint shards."""
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
    cfg = AutoConfig.from_pretrained(model_path)
    try:
        tower = Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg.vision_config, dtype=dtype)
    except TypeError:
        tower = Qwen2_5_VisionTransformerPretrainedModel._from_config(cfg.vision_config, torch_dtype=dtype)
    with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    state = {}
    for shard in sorted({v for k, v in weight_map.items() if k.startswith("visual.")}):
        with safe_open(os.path.join(model_path, shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("visual."):
                    state[k[len("visual."):]] = f.get_tensor(k)
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision tower weights: missing={missing[:5]} unexpected={unexpected[:5]}")
    tower.to(device=device, dtype=dtype).eval()
    for p in tower.parameters():
        p.requires_grad = False
    return tower
@torch.no_grad()
def tower_g(tower, pixel_values, grid_thw):
    """Same call chain as Qwen2_5_VLModel.get_image_features + LVMMemoryFormer's H.mean(0)."""
    dev = next(tower.parameters()).device
    out = tower(pixel_values.to(dev).type(tower.dtype), grid_thw=grid_thw.to(dev))
    feats = out.pooler_output
    return feats.float().mean(dim=0), feats
def write_clip(root, uid, gs, jpgs, sizes, n_max, err, sec):
    os.makedirs(os.path.join(root, "g"), exist_ok=True)
    os.makedirs(os.path.join(root, "frames"), exist_ok=True)
    os.makedirs(os.path.join(root, "meta"), exist_ok=True)
    if err is None:
        g = np.stack(gs).astype(np.float32) if gs else np.zeros((0, 0), np.float32)
        tmp = os.path.join(root, "g", f".{uid}.tmp.npy")
        np.save(tmp, g)
        os.replace(tmp, os.path.join(root, "g", f"{uid}.npy"))
        off = np.zeros(len(jpgs) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(b) for b in jpgs])
        tmp = os.path.join(root, "frames", f".{uid}.tmp")
        with open(tmp, "wb") as f:
            for b in jpgs:
                f.write(b)
        os.replace(tmp, os.path.join(root, "frames", f"{uid}.bin"))
        tmp = os.path.join(root, "frames", f".{uid}.off.tmp.npy")
        np.save(tmp, off)
        os.replace(tmp, os.path.join(root, "frames", f"{uid}.off.npy"))
    meta = {"clip_uid": uid, "n_frames": len(gs), "n_expected": n_max, "sizes": sorted({tuple(s) for s in sizes}),
            "dim": int(gs[0].shape[0]) if gs else None, "frame_format": FRAME_FORMAT, "error": err, "sec": round(sec, 1)}
    tmp = os.path.join(root, "meta", f".{uid}.tmp.json")
    with open(tmp, "w") as f:
        json.dump(meta, f)
    os.replace(tmp, os.path.join(root, "meta", f"{uid}.json"))
    return meta
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", default="train,val", help="test is unannotated and never read downstream")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_clips", type=int, default=None, help="First N clips of this shard (pilot).")
    ap.add_argument("--clips_file", default=None, help="Only these clip_uids (one per line), e.g. the pilot streams' clips.")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    cfg = load_yaml(args.config)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["model"].get("torch_dtype", "float16")]
    sample_fps = float(cfg["data"].get("sample_fps", 1.0))
    max_side = int(cfg["data"].get("frame_max_side", 448))
    model_path = snapshot_download(cfg["model"]["model_name_or_path"], local_files_only=True)
    clips = clip_list(args.clean_dir, args.splits.split(","))
    if args.clips_file:
        wanted = {l.strip() for l in open(args.clips_file) if l.strip()}
        clips = [c for c in clips if c["clip_uid"] in wanted]
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard]
    def done(c):
        path = os.path.join(args.out_dir, "meta", f"{c['clip_uid']}.json")
        if not os.path.exists(path):
            return False
        m = json.load(open(path))
        return not m.get("error") and m.get("frame_format") == FRAME_FORMAT  # older JPEG entries are redone
    todo = [c for c in clips if not done(c)]
    if args.max_clips is not None:
        todo = todo[: args.max_clips]
    print(f"[cache] shard {args.shard}/{args.num_shards}: {len(clips)} clips, {len(todo)} to do, dtype={dtype}, max_side={max_side}, "
          f"GPU={torch.cuda.get_device_name(0)}", flush=True)
    if not todo:
        return
    tower = load_tower(model_path, dtype, "cuda:0")
    print(f"[cache] tower attn={getattr(tower.config, '_attn_implementation', None)} params={sum(p.numel() for p in tower.parameters())/1e6:.0f}M", flush=True)
    loader = DataLoader(ClipFrames(todo, args.video_dir, sample_fps, max_side, model_path), batch_size=None,
                        num_workers=args.workers, prefetch_factor=64 if args.workers else None)
    partial, t_start, n_frames, gpu_sec, n_clips = {}, time.time(), 0, 0.0, 0
    clip_t0 = {}
    for rec in loader:
        if rec[0] == "frame":
            _, uid, k, pv, grid, jpg, size = rec
            p = partial.setdefault(uid, {"g": [], "jpg": [], "sizes": []})
            clip_t0.setdefault(uid, time.time())
            if k != len(p["g"]):
                raise RuntimeError(f"{uid}: frame {k} arrived out of order (have {len(p['g'])})")
            t0 = time.time()
            g, feats = tower_g(tower, pv, grid)
            if not bool(torch.isfinite(feats).all()):
                p["err"] = f"non-finite visual features at frame {k} in {dtype}"
            torch.cuda.synchronize()
            gpu_sec += time.time() - t0
            p["g"].append(g.cpu().numpy())
            p["jpg"].append(jpg)
            p["sizes"].append(size)
            n_frames += 1
        else:
            _, uid, count, err, n_max = rec
            p = partial.pop(uid, {"g": [], "jpg": [], "sizes": []})
            err = err or p.get("err")
            if err is None and count == 0:
                err = "no frames decoded"
            meta = write_clip(args.out_dir, uid, p["g"], p["jpg"], p["sizes"], n_max, err, time.time() - clip_t0.pop(uid, time.time()))
            n_clips += 1
            el = time.time() - t_start
            print(f"[cache] {n_clips}/{len(todo)} {uid} frames={meta['n_frames']}/{n_max} err={err} | total {n_frames} frames "
                  f"{n_frames / max(el, 1e-6):.1f} fps, gpu {1000 * gpu_sec / max(n_frames, 1):.1f} ms/frame, elapsed {el / 60:.1f} min", flush=True)
    print(f"[cache] shard {args.shard} done: {n_clips} clips, {n_frames} frames, {(time.time() - t_start) / 3600:.2f} h", flush=True)
if __name__ == "__main__":
    main()
