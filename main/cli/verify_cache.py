"""Check the Stage-0 cache against the online path of the full model (needs the 4-GPU layout).

For the first --num_clips cached clips, every --stride-th frame:
  g:     cached g_t vs the online g_t = encode_frame(img).float().mean(0)  (max |diff|, min cosine)
  frame: cached JPEG vs the online frame (PSNR), and the effect on the query-time prompt pass
         (relative difference of the last-layer hidden state at the final prompt token).
Exits non-zero if the min cosine of cached vs online g_t is below --min_cos (fp16 kernels differ in the last
bits, so an absolute tolerance is the wrong test), a frame count differs, or the JPEG copy moves the prompt
hidden state by more than --max_jpeg_ratio x the change between two consecutive real frames.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from main.cli.common import load_yaml
from main.cli.ego4d_common import format_prompt, load_model, log_device
from main.data.frame_cache import FrameCache
from main.utils.video import iter_video_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--num_clips", type=int, default=3)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--min_cos", type=float, default=0.9999)
    ap.add_argument("--max_jpeg_ratio", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    fps, max_side = float(cfg["data"]["sample_fps"]), int(cfg["data"]["frame_max_side"])
    log_device()
    model = load_model(cfg)
    model.eval()
    cache = FrameCache(args.cache_dir)
    with open(f"{args.clean_dir}/clips.json") as f:
        clips = [c for c in json.load(f) if cache.has(c["clip_uid"])][: args.num_clips]
    res = {"clips": [], "g_max_abs": 0.0, "g_min_cos": 1.0, "g_max_rel_l2": 0.0, "psnr": [], "hidden_rel": [], "hidden_rel_next": [], "count_mismatch": 0}
    with torch.no_grad():
        for c in clips:
            uid = c["clip_uid"]
            g_cache = cache.g(uid)
            n_max = int(math.ceil(float(c["duration_sec"]) * fps - 1e-6))
            n_online, enc_sec, prev_img = 0, [], None
            for k, (img, _) in enumerate(iter_video_frames(f"{args.video_dir}/{uid}.mp4", sample_fps=fps, max_side=max_side)):
                if k >= n_max:
                    break
                n_online += 1
                if k % args.stride or prev_img is None:
                    prev_img = img
                    continue
                t0 = time.time()
                H, _ = model.encode_frame(img)
                torch.cuda.synchronize()
                enc_sec.append(time.time() - t0)
                g_online = H[0].to(device=model.trainable_device, dtype=torch.float32).mean(0).cpu()
                gc = torch.from_numpy(np.array(g_cache[k], dtype=np.float32))
                res["g_max_abs"] = max(res["g_max_abs"], float((gc - g_online).abs().max()))
                res["g_min_cos"] = min(res["g_min_cos"], float(F.cosine_similarity(gc, g_online, dim=0)))
                res["g_max_rel_l2"] = max(res["g_max_rel_l2"], float((gc - g_online).norm() / g_online.norm()))
                jimg = cache.image(uid, k)
                a, b = np.asarray(img, dtype=np.float64), np.asarray(jimg, dtype=np.float64)
                mse = float(((a - b) ** 2).mean()) if a.shape == b.shape else float("nan")
                res["psnr"].append(10 * math.log10(255.0 ** 2 / max(mse, 1e-12)))
                hs = []
                for im in (img, jimg, prev_img):
                    inputs = model.prepare_inputs([im], [format_prompt("Where did I put the keys?")])
                    out = model.base_model(**inputs, output_hidden_states=True)
                    hs.append(out.hidden_states[-1][0, -1].float().cpu())
                res["hidden_rel"].append(float((hs[0] - hs[1]).norm() / hs[0].norm()))
                res["hidden_rel_next"].append(float((hs[0] - hs[2]).norm() / hs[0].norm()))
                prev_img = img
            if n_online != cache.num_frames(uid):
                res["count_mismatch"] += 1
            res["clips"].append({"clip_uid": uid, "frames_online": n_online, "frames_cache": cache.num_frames(uid),
                                 "encode_frame_ms": 1000 * float(np.mean(enc_sec)) if enc_sec else None})
            print(json.dumps(res["clips"][-1]), flush=True)
    summary = {k: v for k, v in res.items() if k not in ("psnr", "hidden_rel", "hidden_rel_next")}
    summary["psnr_min"], summary["psnr_mean"] = float(np.min(res["psnr"])), float(np.mean(res["psnr"]))
    summary["hidden_rel_max"], summary["hidden_rel_mean"] = float(np.max(res["hidden_rel"])), float(np.mean(res["hidden_rel"]))
    summary["hidden_rel_next_frame_mean"] = float(np.mean(res["hidden_rel_next"]))
    summary["jpeg_vs_next_frame_ratio"] = summary["hidden_rel_mean"] / max(summary["hidden_rel_next_frame_mean"], 1e-9)
    summary["frames_checked"] = len(res["psnr"])
    summary["ok"] = (summary["g_min_cos"] >= args.min_cos and summary["count_mismatch"] == 0
                     and summary["jpeg_vs_next_frame_ratio"] <= args.max_jpeg_ratio)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
