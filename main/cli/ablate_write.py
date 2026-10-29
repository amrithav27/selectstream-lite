"""Offline write-side ablations (docs/plan.md §13.3), CPU only, from the Stage-0 cache.

The same closed-form replay as Check 2 (replay_write.py: Memory, the merge penalty, the read at t_q, best-node
IoU), over a grid of write rules x write signals, every cell at a matched cut budget:

  L_min        --l_mins: Eq. 3 with that minimum segment length (frames = s at 1 fps); 8 is the paper's.
               AB-G2 step 1. At a matched budget the energy cut is off in almost every cell (below), so
               the cuts are where the signal jumps: a smaller L_min already lets segments be short exactly
               where the signal says so. A separate step-2 rule waits for step 1 (docs/plan.md §13.3).
  variant      "<cut signal>[:<value signal>]"; the value signal (default: the cut signal) is the one the
               update-vs-new rule (segment mean < tau_s) and the merge penalty p_sup read (AB-U1).
               Signals: A = sbar (paper), As = raw s, C = EMA(y) (oracle), Cr = raw y, D = fixed length,
               c4 = causal boundary score 1 - cos(mean g[t-7..t-4], mean g[t-3..t]) (GEBD-style, AB-G1),
               B.<name> = EMA(sigmoid(scorer)) for --scorers name=path (AB-I1/L1/L2), Br.<name> = raw,
               E = (sbar / q99(sbar) + B.<first>) / 2 (AB-I2).
  --norm_value replaces the value signal by its causal rank among the last 64 frames (AB-L2: tau_s and
               p_sup then read a scale-free signal, so a per-video offset cannot decide update vs new).
  --target     fixed: every cell writes as many segments per stream as arm A at L_min = 8 (granularity is
               redistributed, not added); per_lmin: as many as arm A at the same L_min.

Calibration on train-fit streams as Check 2: B_s (or the spike quantile) bisected per cell to the target.
On an EMA-smoothed signal every frame of a rise is its window's maximum, so with S < 8 the jump test fires on
each rising frame whatever the quantile and the quantile cannot reach the target. Those cells instead bisect a
margin m (paper quantile kept): jump if w_t > theta_t + m * sd(last 64 w). m = 0 is Eq. 3.
tau_s = median segment-mean value signal; tau_r = 0.9 quantile of arm A's nearest-node cosines (L_min = 8,
N main), shared by every cell; D's L = stream frames / target.
Reported per track, rule, N and variant: IoU* (open segment included) and committed IoU*, survival,
nodes, mean span, segments, merges, updates; paired bootstrap (over videos) differences vs A at the same
L_min and vs A at L_min = 8; within-video / local AUC of every signal against y; answer-window lengths.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from main.cli.common import load_yaml
from main.cli.ego4d_common import frame_labels, train_participants
from main.cli.replay_write import bootstrap_diff, load_streams, recent_quantile, replay_stream, seg_means
from main.cli.train_scorer import auroc, ema, surprise, trailing_mean, within_auc
from main.data.ego4d_qa import Ego4DStreamDataset
from main.data.frame_cache import FrameCache
from main.model.importance_scorer import load_scorer

REF_RULE = 8


def cut_points_lmin(w, gap_cut, l_min, l_max, b_s, theta):
    """Eq. 3 (replay_write.cut_points) with a precomputed threshold theta_t."""
    cuts, seg_len, energy = [], 0, 0.0
    for i in range(len(w)):
        seg_len += 1
        energy += w[i]
        cut = seg_len >= l_min and (w[i] > theta[i] or energy > b_s or seg_len >= l_max)
        if gap_cut[i] or cut:
            cuts.append(i)
            seg_len, energy = 0, 0.0
    return cuts


def fixed_cuts(n, gap_cut, L):
    cuts, seg_len = [], 0
    for i in range(n):
        seg_len += 1
        if seg_len >= L or gap_cut[i]:
            cuts.append(i)
            seg_len = 0
    return cuts


def recent_rank(x, window):
    """Fraction of the last `window` values (current included) that are <= x_t: a causal, scale-free copy."""
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = float(np.mean(x[lo : i + 1] <= x[i]))
    return out


def causal_boundary(g, k=4):
    gu = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-8)
    c = np.concatenate([np.zeros((1, g.shape[1])), np.cumsum(gu, 0)])
    out = np.zeros(len(g))
    for t in range(2 * k - 1, len(g)):
        a = c[t - k + 1] - c[t - 2 * k + 1]
        b = c[t + 1] - c[t - k + 1]
        out[t] = 1.0 - float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return out


class Streams:
    """Signals per stream, keyed by name; `theta` caches the recent-quantile thresholds."""

    def __init__(self, streams, signals, scorers, rho, window, sbar_q99, norm_value):
        self.s = streams
        self.window = window
        for st in self.s:
            g = torch.from_numpy(st["g"])
            s_raw, sbar = surprise(g, rho)
            st["sig"] = {"A": st["A"], "As": s_raw, "C": st["C"], "Cr": st["y"], "D": np.zeros(len(st["t"]))}
            if "c4" in signals:
                st["sig"]["c4"] = causal_boundary(st["g"])
            sc = torch.from_numpy(np.stack([s_raw, sbar], 1).astype(np.float32))
            for name, m in scorers.items():
                with torch.no_grad():
                    a = torch.sigmoid(m.forward_batch(sc, g.float())).numpy().astype(np.float64)
                st["sig"][f"Br.{name}"], st["sig"][f"B.{name}"] = a, ema(a, rho)
            if scorers:
                st["sig"]["E"] = 0.5 * (sbar / sbar_q99 + st["sig"][f"B.{next(iter(scorers))}"])
            st["val"] = {k: (recent_rank(v, window) if norm_value and k != "D" else v) for k, v in st["sig"].items()}
            st["theta"] = {}

    def theta(self, st, key, q, m=0.0):
        k = (key, round(q, 6))
        if k not in st["theta"]:
            st["theta"][k] = recent_quantile(st["sig"][key], q, self.window)
        if m == 0.0:
            return st["theta"][k]
        if (key, "sd") not in st["theta"]:
            st["theta"][(key, "sd")] = recent_sd(st["sig"][key], self.window)
        return st["theta"][k] + m * st["theta"][(key, "sd")]


def recent_sd(x, window):
    c1, c2 = np.concatenate([[0.0], np.cumsum(x)]), np.concatenate([[0.0], np.cumsum(x * x)])
    i = np.arange(len(x))
    lo = np.maximum(i - window + 1, 0)
    n = i + 1 - lo
    mean = (c1[i + 1] - c1[lo]) / n
    return np.sqrt(np.maximum((c2[i + 1] - c2[lo]) / n - mean * mean, 0.0))


def segments(streams, key, rule, P, b_s, q, L_fixed=None, m=0.0):
    out = []
    for st in streams.s:
        if key == "D":
            out.append(fixed_cuts(len(st["t"]), st["cut"], L_fixed))
        else:
            out.append(cut_points_lmin(st["sig"][key], st["cut"], rule, P["l_max"], b_s, streams.theta(st, key, q, m)))
    return out


def match(streams, key, rule, target, P, tol=0.03):
    """(b_s, quantile, margin, segments, note). B_s bisection; the spike quantile raised when jump cuts alone
    exceed the target (as match_budget); the margin when the quantile cannot get within `tol` of it."""
    def n(b, q, m=0.0):
        return float(np.mean([len(c) for c in segments(streams, key, rule, P, b, q, m=m)]))

    lo, hi, q0 = 1e-4, 1e4, P["quantile"]
    if n(hi, q0) > target:
        q_lo, q_hi = q0, 0.999
        for _ in range(16):
            mid = 0.5 * (q_lo + q_hi)
            q_lo, q_hi = (mid, q_hi) if n(hi, mid) > target else (q_lo, mid)
        got = n(hi, q_hi)
        if abs(got - target) <= tol * target:
            return hi, q_hi, 0.0, got, "spike quantile raised (energy budget off)"
        m_lo, m_hi = 0.0, 20.0
        for _ in range(20):
            mid = 0.5 * (m_lo + m_hi)
            m_lo, m_hi = (mid, m_hi) if n(hi, q0, mid) > target else (m_lo, mid)
        got = n(hi, q0, m_hi)
        return hi, q0, m_hi, got, f"spike margin {m_hi:.3f} sd (quantile could not match: {n(hi, q_hi):.1f})"
    for _ in range(24):
        mid = (lo * hi) ** 0.5
        lo, hi = (mid, hi) if n(mid, q0) > target else (lo, mid)
    return hi, q0, 0.0, n(hi, q0), "ok"


def split_variant(v):
    cut, _, value = v.partition(":")
    return cut, value or cut


# Workers read these module globals after fork (no pickling of the streams).
_G = {}


def _calibrate(job):
    rule, variant = job
    train, P, targets = _G["train"], _G["P"], _G["targets"]
    cut, value = split_variant(variant)
    target = targets[rule]
    if (rule, variant) == (REF_RULE, "A"):
        # The reference cell (L_min = 8, arm A) is Check 2's arm A: the paper's B_s and spike quantile, not re-matched.
        cuts = segments(train, "A", rule, P, P["b_s"], P["quantile"])
        c = {"b_s": P["b_s"], "quantile": P["quantile"], "margin": 0.0, "note": "paper values (reference)"}
    elif cut == "D":
        L = max(1, int(round(np.mean([len(st["t"]) for st in train.s]) / target)))
        cuts = segments(train, "D", rule, P, 0, 0, L)
        c = {"fixed_len": L, "b_s": None, "quantile": None, "margin": 0.0, "note": "fixed"}
    else:
        b_s, q, m, _, note = match(train, cut, rule, target, P)
        cuts = segments(train, cut, rule, P, b_s, q, m=m)
        c = {"b_s": b_s, "quantile": q, "margin": m, "note": note}
    c["segments"] = float(np.mean([len(x) for x in cuts]))
    c["tau_s"] = float("inf") if value == "D" else float(np.median([m for st, x in zip(train.s, cuts) for m in seg_means(st["val"][value], x)]))
    return (rule, variant), c


def _replay(job):
    track, rule, variant, N = job
    val, P, cal, tau_r = _G["val"][track], _G["P"], _G["cal"], _G["tau_r"]
    cut, value = split_variant(variant)
    c = cal[(rule, variant)]
    allcuts = segments(val, cut, rule, P, c["b_s"], c["quantile"], c.get("fixed_len"), m=c["margin"])
    rows, merges, updates = [], 0, 0
    for st, cuts in zip(val.s, allcuts):
        r, mem = replay_stream({**st, "_v": st["val"][value]}, "_v", cuts, N, tau_r, c["tau_s"], st["item"].queries)
        for x in r:
            x["segments"] = len(cuts)
        rows += r
        merges += mem.n_merge
        updates += mem.n_update
    return (track, rule, variant, N), rows, merges / max(len(val.s), 1), updates / max(len(val.s), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scorers", default="", help="name=path,... Stage-1 scorer.pt files (train_scorer.py); signals B.<name>, Br.<name>.")
    ap.add_argument("--variants", default="A,C,D", help="Comma list of <cut>[:<value>] (see the module docstring).")
    ap.add_argument("--l_mins", default="8,4,2", help="Comma list of minimum segment lengths (frames).")
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--target", choices=("fixed", "per_lmin"), default="fixed")
    ap.add_argument("--norm_value", action="store_true")
    ap.add_argument("--tracks", default="nlq,qa")
    ap.add_argument("--max_streams", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    v, dcfg, tcfg = cfg["vismem"], cfg["data"], cfg["training"]
    fps, rho, tol, seed = float(dcfg["sample_fps"]), float(v["lvm_saw_ema_decay"]), float(tcfg["time_tolerance"]), int(tcfg["seed"])
    P = {"l_max": int(v["lvm_saw_l_max"]), "quantile": float(v["lvm_saw_quantile"]), "window": int(v["lvm_saw_recent_window"]), "b_s": float(v["lvm_saw_energy_budget"])}
    if int(v["lvm_saw_l_min"]) != REF_RULE:
        raise ValueError(f"config L_min {v['lvm_saw_l_min']} != reference rule {REF_RULE}")
    n_main = int(v["lvm_num_slots"])
    variants = [x for x in args.variants.split(",") if x]
    rules = [int(x) for x in args.l_mins.split(",") if x]
    if REF_RULE not in rules:
        rules = [REF_RULE] + rules
    if "A" not in variants:
        variants = ["A"] + variants
    budgets = [int(x) for x in args.budgets.split(",")]
    scorers = {}
    for kv in [x for x in args.scorers.split(",") if x]:
        name, path = kv.split("=", 1)
        scorers[name] = load_scorer(path)
    signals = {s for var in variants for s in split_variant(var)}
    missing = {s for s in signals if s.startswith(("B.", "Br.")) and s.split(".", 1)[1] not in scorers} | ({"E"} & signals if not scorers else set())
    if missing:
        raise ValueError(f"signals need --scorers: {sorted(missing)}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(1)
    cache = FrameCache(args.feature_cache)
    seen = train_participants(args.clean_dir)

    def ds(split, source, part):
        d = Ego4DStreamDataset(args.clean_dir, "", split, source, part=part, dev_frac=float(dcfg["dev_frac"]), sample_fps=fps, seed=seed,
                               require_video=False, query_time_mode=str(dcfg.get("query_time_mode", "uniform")))
        d.items = [i for i in d.items if all(cache.has(u) for u in i.clip_uids)][: args.max_streams]
        d.items = [d[i] for i in range(len(d.items))]  # fixed (epoch-0) ask times, as Check 2
        return d

    def prepare(streams):
        for st in streams:
            st["y"] = frame_labels(st["t"], [(q.window_start, q.window_end) for q in st["item"].queries], tol)
        return streams

    t0 = time.time()
    raw_train = prepare(load_streams(ds("train", "merged", "train"), cache, fps, rho, tol, None, None))
    sbar_q99 = float(np.quantile(np.concatenate([st["A"] for st in raw_train]), 0.99))
    train = Streams(raw_train, signals, scorers, rho, P["window"], sbar_q99, args.norm_value)
    print(f"[ablate] train-fit streams {len(train.s)} loaded ({time.time() - t0:.0f}s); signals {sorted(signals)}", flush=True)

    # Targets: arm A's segments per stream at L_min = 8 (fixed) or at each L_min (per_lmin).
    targets = {}
    for rule in rules:
        r = REF_RULE if args.target == "fixed" else rule
        if r not in targets:
            targets[r] = float(np.mean([len(c) for c in segments(train, "A", r, P, P["b_s"], P["quantile"])]))
        targets[rule] = targets[r]
    # tau_r from arm A's own writes at L_min = 8 at the main budget (as Check 2), shared by all cells.
    near = []
    for st, c in zip(train.s, segments(train, "A", REF_RULE, P, P["b_s"], P["quantile"])):
        _, mem = replay_stream({**st, "_v": st["A"]}, "_v", c, n_main, tau_r=2.0, tau_s=-1.0, queries=[])
        near += mem.near_cos
    tau_r = float(np.quantile(near, 0.9))
    print(f"[ablate] targets {targets}, tau_r {tau_r:.4f} ({time.time() - t0:.0f}s)", flush=True)

    _G.update(train=train, P=P, targets=targets)
    jobs = [(rule, var) for rule in rules for var in variants]
    with mp.get_context("fork").Pool(args.workers) as pool:
        cal = dict(pool.map(_calibrate, jobs))
    for k, c in cal.items():
        print(f"[ablate] cal {k}: {json.dumps(c)}", flush=True)
    del train, raw_train
    _G.pop("train")

    results = {"args": vars(args), "calibration": {f"Lmin={r}|{v_}": c for (r, v_), c in cal.items()}, "tau_r": tau_r, "targets": targets, "tracks": {}}
    for track in [t for t in args.tracks.split(",") if t]:
        src = {"nlq": "nlq", "qa": "qaego4d"}[track]
        raw = prepare(load_streams(ds("val", src, "all"), cache, fps, rho, tol, None, seen))
        val = Streams(raw, signals, scorers, rho, P["window"], sbar_q99, args.norm_value)
        ys = [st["y"] for st in val.s]
        sig_auc = {}
        for s in sorted(signals - {"D"}):
            xs = [st["sig"][s] for st in val.s]
            y = np.concatenate(ys)
            sig_auc[s] = {"within_auc": within_auc(xs, ys), "local_auc": auroc(np.concatenate([a - trailing_mean(a, P["window"]) for a in xs]), y),
                          "pooled_auc": auroc(np.concatenate(xs), y)}
        wl = np.array([q.window_end - q.window_start for st in val.s for q in st["item"].queries])
        _G["val"] = {track: val}
        _G.update(cal=cal, tau_r=tau_r)
        rjobs = [(track, rule, var, N) for rule in rules for var in variants for N in budgets]
        with mp.get_context("fork").Pool(args.workers) as pool:
            out = pool.map(_replay, rjobs)
        per = {k: (rows, m, u) for k, rows, m, u in out}
        cells = {}
        for (tr, rule, var, N), (rows, m, u) in per.items():
            e = {k: float(np.mean([r[k] for r in rows])) for k in ("best_iou", "best_iou_committed", "survival", "num_nodes", "mean_span_s", "segments")}
            e.update(queries=len(rows), merges_per_stream=m, updates_per_stream=u,
                     disjoint_best_iou=float(np.mean([r["best_iou"] for r in rows if r["disjoint"]])) if any(r["disjoint"] for r in rows) else None)
            if var != "A":
                e["vs_A_same_lmin"] = {k: bootstrap_diff(per[(tr, rule, "A", N)][0], rows, k) for k in ("best_iou", "best_iou_committed")}
            if rule != REF_RULE or var != "A":
                e["vs_A_lmin8"] = {k: bootstrap_diff(per[(tr, REF_RULE, "A", N)][0], rows, k) for k in ("best_iou", "best_iou_committed")}
            cells[f"Lmin={rule}|{var}|N={N}"] = e
            d1 = e.get("vs_A_same_lmin", {}).get("best_iou")
            d2 = e.get("vs_A_lmin8", {}).get("best_iou")
            fmt = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]" if d else "      --        "
            print(f"[ablate] {track} Lmin={rule:<2d} N={N:<3d} {var:14s} IoU* {e['best_iou']:.3f} c {e['best_iou_committed']:.3f} seg {e['segments']:5.1f} "
                  f"nodes {e['num_nodes']:4.1f} span {e['mean_span_s']:4.0f}s merges {m:5.1f} upd {u:5.1f} | vs A same L_min {fmt(d1)} | vs A L_min 8 {fmt(d2)}", flush=True)
        results["tracks"][track] = {"cells": cells, "signal_auc": sig_auc, "streams": len(val.s),
                                    "window_len_s": {"q10_25_50_75_90": np.quantile(wl, [.1, .25, .5, .75, .9]).tolist(), "frac_below": {str(k): float((wl < k).mean()) for k in (2, 4, 8)}}}
        for s, a in sig_auc.items():
            print(f"[ablate] {track} signal {s:12s} within {a['within_auc']:.3f} local {a['local_auc']:.3f} pooled {a['pooled_auc']:.3f}", flush=True)
        del val, raw
    with open(os.path.join(args.out_dir, "ablate_write.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"[ablate] wrote {args.out_dir}/ablate_write.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
