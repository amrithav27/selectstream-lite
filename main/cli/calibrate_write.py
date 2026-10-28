"""Calibrate per-arm write thresholds so every arm makes the same number of cuts (docs/plan.md §3).

One no-grad pass over a fixed subset of train clips records, per frame, the three write signals
(A: sbar_t, B: EMA of the Stage-R scorer's a_t, C: EMA of y_t). w_t depends on the frames only, so
Eq. 3 can then be replayed offline for any threshold. For B and C we pick:
  B_s  so that mean segments/clip equals arm A's (bisection; segments are non-increasing in B_s),
  tau  so that the fraction of segments with mean w below tau equals arm A's fraction below tau_s.
Arm D gets L = round(arm A's mean segment length). Prints a YAML `arms:` block.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from main.cli.common import load_yaml
from main.cli.ego4d_common import apply_arm, frame_labels, load_model, log_device, prepare_scorer
from main.cli.replay_write import cut_points, match_budget as match_budget_cuts, seg_means
from main.data.ego4d_qa import Ego4DStreamDataset
from main.data.frame_cache import FrameCache, stream_inputs


TAU_S_QUANTILE = 0.5  # half of all segments may update an existing node (if similar enough)
TAU_R_QUANTILE = 0.9  # the 10% of segments most similar to a stored node (frozen g space) may update it

def replay_eq3(ws, l_min, l_max, b_s, quantile, window):
    """Exact offline copy of the boundary rule in LVMMemoryFormer._update_temporal_saw."""
    recent, seg, segs, energy = [], [], [], 0.0
    for i, w in enumerate(ws):
        recent = (recent + [w])[-window:]
        seg.append(w)
        energy += w
        theta = float(np.quantile(recent, quantile))
        cut = len(seg) >= l_min and (w > theta or energy > b_s or len(seg) >= l_max)
        if i == len(ws) - 1:
            cut = True
        if cut:
            segs.append(float(np.mean(seg)))
            seg, energy = [], 0.0
    return segs


def ema(raw, rho):
    out, prev = [], 0.0
    for x in raw:
        prev = rho * prev + (1.0 - rho) * x
        out.append(prev)
    return out


def match_budget(seqs, target_segs, l_min, l_max, quantile, window):
    def mean_segs(b):
        return float(np.mean([len(replay_eq3(s, l_min, l_max, b, quantile, window)) for s in seqs]))

    lo, hi = 1e-4, 1e4
    if mean_segs(hi) > target_segs:
        # Spike cuts alone (w_t above its own recent quantile) already give more segments than arm A,
        # so no energy budget can match it (job 17646: 34 vs 25.3 with an untrained scorer). Switch the
        # energy rule off and raise the spike quantile instead: at q = 1 no frame can exceed the max of
        # a window it belongs to, leaving only L_max cuts, so the count is matched somewhere in [q, 1].
        def segs_q(q):
            return float(np.mean([len(replay_eq3(s, l_min, l_max, hi, q, window)) for s in seqs]))
        q_lo, q_hi = quantile, 1.0
        for _ in range(30):
            mid = 0.5 * (q_lo + q_hi)
            if segs_q(mid) > target_segs:
                q_lo = mid
            else:
                q_hi = mid
        return hi, segs_q(q_hi), "spike quantile raised (energy budget off)", q_hi
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        if mean_segs(mid) > target_segs:
            lo = mid
        else:
            hi = mid
    return hi, mean_segs(hi), "ok", quantile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--torch_dtype", default=None, help="Override model.torch_dtype (bfloat16 | float16 | float32).")
    ap.add_argument("--ckpt", default=None, help="Stage-R checkpoint (the writer whose node cosines set tau_r).")
    ap.add_argument("--scorer", default=None, help="Stage-1 scorer for arm B; without it arm B is not calibrated.")
    ap.add_argument("--feature_cache", default=None, help="Stage-0 cache: no backbone pass per frame.")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--num_clips", type=int, default=None, help="Number of train-fit streams (default: all).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg_dict = apply_arm(load_yaml(args.config), "A")
    if args.torch_dtype:
        cfg_dict["model"]["torch_dtype"] = args.torch_dtype
    v, dcfg, tcfg = cfg_dict["vismem"], cfg_dict["data"], cfg_dict["training"]
    log_device()
    model = load_model(cfg_dict, ckpt=args.ckpt)
    prepare_scorer(model, "A", args.scorer)
    model.eval()
    cache = FrameCache(args.feature_cache) if args.feature_cache else None
    sample_fps = float(dcfg.get("sample_fps", 1.0))
    ds = Ego4DStreamDataset(args.clean_dir, args.video_dir, "train", "merged", part="train", sample_fps=sample_fps,
                            seed=int(tcfg.get("seed", 42)), require_video=cache is None)
    if cache is not None:
        ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
    rng = np.random.default_rng(int(tcfg.get("seed", 42)))
    n = len(ds) if args.num_clips is None else min(args.num_clips, len(ds))
    picked = sorted(rng.choice(len(ds), size=n, replace=False).tolist())
    rho, tol = float(v["lvm_saw_ema_decay"]), float(tcfg.get("time_tolerance", 0.5))
    P = {"l_min": int(v["lvm_saw_l_min"]), "l_max": int(v["lvm_saw_l_max"]), "quantile": float(v["lvm_saw_quantile"]),
         "window": int(v["lvm_saw_recent_window"])}
    b_s_a = float(v["lvm_saw_energy_budget"])

    streams = []
    cos_a = []  # arm A: cosine of each new segment's mean g to the nearest stored node's mean g
    with torch.no_grad():
        for k, idx in enumerate(picked):
            item = ds.items[idx]
            state = model.init_memory_state(batch_size=1, dtype=model.trainable_dtype, device=model.trainable_device)
            windows = [(q.window_start, q.window_end) for q in item.queries]
            s_ema, a, y, cuts = [], [], [], []
            for img, g, t, cut, _ in stream_inputs(item, sample_fps, int(dcfg.get("frame_max_side", 448)), cache):
                state, info = model.update_memory_frame(img, t, state, flush=cut, g=g)
                s_ema.append(info["s_ema"])
                if info.get("write") is not None and info["write"].get("best_sim") is not None:
                    cos_a.append(info["write"]["best_sim"])
                a.append(float(torch.sigmoid(info["importance_logit"])) if info.get("importance_logit") is not None else 0.0)
                y.append(float(frame_labels([t], windows, tol)[0]))
                cuts.append(bool(cut))
            streams.append({"A": np.array(s_ema), "B": np.array(ema(a, rho)), "C": np.array(ema(y, rho)), "cut": np.array(cuts)})
            if (k + 1) % 25 == 0 or k + 1 == n:
                print(f"[calib] stream {k + 1}/{n} {item.stream_id} frames={len(s_ema)}", flush=True)

    cuts_a = [cut_points(s["A"], s["cut"], P["l_min"], P["l_max"], b_s_a, P["quantile"], P["window"]) for s in streams]
    target = float(np.mean([len(c) for c in cuts_a]))
    # tau_s (update-vs-new) = the TAU_S_QUANTILE quantile of each arm's segment-mean signal, so every arm may
    # update the same fraction of segments; tau_r = TAU_R_QUANTILE quantile of arm A's nearest-node cosines.
    seg_means_a = [m for s, c in zip(streams, cuts_a) for m in seg_means(s["A"], c)]
    tau_s_a = float(np.quantile(seg_means_a, TAU_S_QUANTILE))
    tau_r = float(np.quantile(cos_a, TAU_R_QUANTILE)) if cos_a else float(v["lvm_tau_r"])
    mean_len = float(np.mean([len(s["A"]) / max(1, len(c)) for s, c in zip(streams, cuts_a)]))
    result = {
        "streams": [ds.items[i].stream_id for i in picked],
        "A": {"segments_per_stream": target, "frac_below_tau": TAU_S_QUANTILE, "mean_segment_len": mean_len,
              "lvm_saw_energy_budget": b_s_a, "lvm_tau_s": tau_s_a, "lvm_tau_r": tau_r,
              "nearest_cos_quantiles": {str(q): float(np.quantile(cos_a, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if cos_a else None},
        "D": {"lvm_fixed_segment_len": int(round(mean_len)), "lvm_tau_r": tau_r},
    }
    for arm in ("B", "C"):
        if arm == "B" and not args.scorer:
            print("[calib][WARN] no --scorer: arm B not calibrated")
            continue
        b_s, q_arm, got, note = match_budget_cuts(streams, arm, target, P)
        segmeans = [m for s in streams for m in seg_means(s[arm], cut_points(s[arm], s["cut"], P["l_min"], P["l_max"], b_s, q_arm, P["window"]))]
        result[arm] = {"lvm_saw_energy_budget": b_s, "lvm_saw_quantile": q_arm, "segments_per_stream": got, "note": note, "lvm_tau_r": tau_r,
                       "lvm_tau_s": float(np.quantile(segmeans, TAU_S_QUANTILE)) if segmeans else None}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "streams"}, indent=1))


if __name__ == "__main__":
    main()
