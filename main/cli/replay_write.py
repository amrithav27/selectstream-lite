"""Check 2 (docs/plan.md §4): is there headroom on the write side at all? CPU only, from the Stage-0 cache.

Replays the memory's write rules for arms A (surprise), B (Stage-1 scorer), C (oracle labels) and D (fixed
length) at a matched cut budget, and measures at every eval ask time whether the needed evidence is still in
memory. Node states are closed-form (each node's mean frozen g_t stands in for the trained slot h), so this
estimates what the write *signal* can change, not the trained system's numbers.

Rules mirrored from LVMMemoryFormer (lvm_memory.py):
  cut      Eq. 3 on w_t: seg_len >= L_min and (w_t > Q_q(last 64 w) or energy > B_s or seg_len >= L_max); arm D:
           seg_len >= L; every arm also cuts at stream gaps and at the end (stream_frames' `cut`)
  write    nearest node by cos(mean g of segment, node mean g); update if cos > tau_r and mean w < tau_s, else new
  merge    when |M| > N: argmin over pairs of 1*(1-(cos+1)/2) + 0.5*mean(w_hat) + 0.25*0 (no reads here)
           + 0.25*mean(recency_hat); the node with more writes is kept; spans union
  read     at t_q: committed nodes + the open segment as a provisional node
Calibration (train-fit streams, the same procedure as calibrate_write): B_s (or the spike quantile) per arm so
mean segments per stream equal arm A's; tau_s = median segment-mean w per arm; tau_r = 0.9 quantile of arm A's
nearest-node cosines; D's L = arm A's mean segment length.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from main.cli.common import load_yaml
from main.cli.ego4d_common import delay_bucket, frame_labels, train_participants
from main.cli.train_scorer import ema, surprise
from main.data.ego4d_qa import Ego4DStreamDataset
from main.data.frame_cache import FrameCache
from main.model.importance_scorer import ImportanceScorer


def recent_quantile(w: np.ndarray, q: float, window: int) -> np.ndarray:
    """theta_t = quantile of w[t-window+1 .. t] (current value included), as `recent_ema` online."""
    out = np.empty(len(w))
    head = min(window - 1, len(w))
    for i in range(head):
        out[i] = np.quantile(w[: i + 1], q)
    if len(w) >= window:
        out[window - 1 :] = np.quantile(np.lib.stride_tricks.sliding_window_view(w, window), q, axis=1)
    return out


def cut_points(w, gap_cut, l_min, l_max, b_s, quantile, window, fixed_len=None):
    """Indices i where the open segment is closed after frame i (online Eq. 3 + forced cuts)."""
    theta = recent_quantile(w, quantile, window) if fixed_len is None else None
    cuts, seg_len, energy = [], 0, 0.0
    for i in range(len(w)):
        seg_len += 1
        energy += w[i]
        if fixed_len is not None:
            cut = seg_len >= fixed_len
        else:
            cut = seg_len >= l_min and (w[i] > theta[i] or energy > b_s or seg_len >= l_max)
        if gap_cut[i]:
            cut = True
        if cut:
            cuts.append(i)
            seg_len, energy = 0, 0.0
    return cuts


def seg_means(w, cuts):
    out, a = [], 0
    for c in cuts:
        out.append(float(np.mean(w[a : c + 1])))
        a = c + 1
    return out


def match_budget(streams, key, target, P):
    """B_s (bisection) so mean segments/stream == target; spike quantile raised if spikes alone exceed it."""
    def segs(b, q):
        return float(np.mean([len(cut_points(s[key], s["cut"], P["l_min"], P["l_max"], b, q, P["window"])) for s in streams]))

    lo, hi = 1e-4, 1e4
    if segs(hi, P["quantile"]) > target:
        q_lo, q_hi = P["quantile"], 1.0
        for _ in range(20):
            mid = 0.5 * (q_lo + q_hi)
            if segs(hi, mid) > target:
                q_lo = mid
            else:
                q_hi = mid
        return hi, q_hi, segs(hi, q_hi), "spike quantile raised (energy budget off)"
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        if segs(mid, P["quantile"]) > target:
            lo = mid
        else:
            hi = mid
    return hi, P["quantile"], segs(hi, P["quantile"]), "ok"


class Memory:
    def __init__(self, N, tau_r, tau_s, dim):
        self.N, self.tau_r, self.tau_s = N, tau_r, tau_s
        self.g = np.zeros((0, dim))
        self.cnt = np.zeros(0)
        self.t0 = np.zeros(0)
        self.t1 = np.zeros(0)
        self.w = np.zeros(0)
        self.wc = np.zeros(0)
        self.near_cos = []
        self.n_update = self.n_merge = 0

    def write(self, gseg, n, wbar, ts, te):
        if len(self.cnt):
            gn = self.g / np.linalg.norm(self.g, axis=1, keepdims=True).clip(1e-12)
            cos = gn @ (gseg / max(np.linalg.norm(gseg), 1e-12))
            j = int(np.argmax(cos))
            self.near_cos.append(float(cos[j]))
            if cos[j] > self.tau_r and wbar < self.tau_s:
                c = self.cnt[j]
                self.g[j] = (c * self.g[j] + n * gseg) / (c + n)
                self.cnt[j] = c + n
                self.t0[j], self.t1[j] = min(self.t0[j], ts), max(self.t1[j], te)
                self.wc[j] += 1
                self.w[j] += (wbar - self.w[j]) / self.wc[j]
                self.n_update += 1
                return
        self.g = np.vstack([self.g, gseg[None]])
        self.cnt = np.append(self.cnt, n)
        self.t0, self.t1 = np.append(self.t0, ts), np.append(self.t1, te)
        self.w, self.wc = np.append(self.w, wbar), np.append(self.wc, 1.0)
        if len(self.cnt) > self.N:
            self.merge()

    def merge(self):
        gn = self.g / np.linalg.norm(self.g, axis=1, keepdims=True).clip(1e-12)
        sim = gn @ gn.T
        wn = self.w / self.w.max() if self.w.max() > 0 else self.w
        r = self.t1 - self.t1.min()
        rn = r / r.max() if r.max() > 0 else r
        pen = 1.0 * (1 - 0.5 * (sim + 1)) + 0.5 * 0.5 * (wn[:, None] + wn[None]) + 0.25 * 0.5 * (rn[:, None] + rn[None])
        pen[np.tril_indices(len(pen))] = np.inf
        k, f = np.unravel_index(np.argmin(pen), pen.shape)
        if self.wc[f] > self.wc[k]:
            k, f = f, k
        ck, cf = self.cnt[k], self.cnt[f]
        self.g[k] = (ck * self.g[k] + cf * self.g[f]) / (ck + cf)
        wk, wf = max(self.wc[k], 1), max(self.wc[f], 1)
        self.w[k] = (wk * self.w[k] + wf * self.w[f]) / (wk + wf)
        self.cnt[k], self.wc[k] = ck + cf, self.wc[k] + self.wc[f]
        self.t0[k], self.t1[k] = min(self.t0[k], self.t0[f]), max(self.t1[k], self.t1[f])
        keep = np.ones(len(self.cnt), bool)
        keep[f] = False
        for a in ("g", "cnt", "t0", "t1", "w", "wc"):
            setattr(self, a, getattr(self, a)[keep])
        self.n_merge += 1


def overlap_iou(spans, window, half=0.5):
    if len(spans) == 0:
        return 0.0, 0.0
    sp = np.asarray(spans, dtype=np.float64)
    lo, hi = sp[:, 0] - half, sp[:, 1] + half
    ws, we = window[0] - half, window[1] + half
    inter = np.clip(np.minimum(hi, we) - np.maximum(lo, ws), 0, None)
    union = (hi - lo) + (we - ws) - inter
    return float((inter > 0).any()), float((inter / np.clip(union, 1e-6, None)).max())


def replay_stream(st, key, cuts, N, tau_r, tau_s, queries):
    """Write the stream with the given cuts; answer each query at its ask time. Returns per-query rows + memory."""
    g, t, w = st["g"], st["t"], st[key]
    mem = Memory(N, tau_r, tau_s, g.shape[1])
    cutset, a, rows = set(cuts), 0, []
    pending = sorted(queries, key=lambda q: q.ask_time)
    for i in range(len(t)):
        if i in cutset:
            mem.write(g[a : i + 1].mean(0), i + 1 - a, float(np.mean(w[a : i + 1])), t[a], t[i])
            a = i + 1
        while pending and (pending[0].ask_time <= t[i] + 1e-6 or i == len(t) - 1):
            q = pending.pop(0)
            committed = list(zip(mem.t0, mem.t1))
            opened = [(t[a], t[i])] if a <= i else []
            win = (q.window_start, q.window_end)
            surv_c, iou_c = overlap_iou(committed, win)
            surv, iou = overlap_iou(committed + opened, win)
            rows.append({"query_id": q.query_id, "stream_id": st["id"], "delay_bucket": delay_bucket(q), "disjoint": st["disjoint"],
                         "survival": surv, "best_iou": iou, "survival_committed": surv_c, "best_iou_committed": iou_c,
                         "num_nodes": len(mem.cnt), "mean_span_s": float(np.mean(mem.t1 - mem.t0 + 1)) if len(mem.cnt) else 0.0})
    return rows, mem


def load_streams(ds, cache, fps, rho, tol, scorer, seen):
    out = []
    for item in ds.items:
        g, t, cut = cache.stream_g(item, fps)
        if len(t) == 0:
            continue
        gt = torch.from_numpy(g)
        s, sbar = surprise(gt, rho)
        st = {"id": item.stream_id, "item": item, "g": g, "t": t, "cut": cut, "A": sbar,
              "disjoint": seen is not None and item.participant_id is not None and item.participant_id not in seen}
        y = frame_labels(t, [(q.window_start, q.window_end) for q in item.queries], tol)
        st["C"] = ema(y, rho)
        if scorer is not None:
            with torch.no_grad():
                sc = torch.from_numpy(np.stack([s, sbar], 1).astype(np.float32))
                st["B"] = ema(torch.sigmoid(scorer.forward_batch(sc, gt.float())).numpy().astype(np.float64), rho)
        st["D"] = np.zeros(len(t))
        out.append(st)
    return out


def bootstrap_diff(rows_a, rows_b, key, n_boot=1000, seed=0):
    """Paired (by query) mean difference b - a, 95% CI from resampling streams."""
    by = defaultdict(list)
    ra = {r["query_id"]: r for r in rows_a}
    for r in rows_b:
        if r["query_id"] in ra:
            by[r["stream_id"]].append(r[key] - ra[r["query_id"]][key])
    sids = list(by)
    if not sids:
        return None
    sums = np.array([np.sum(by[s]) for s in sids])
    cnts = np.array([len(by[s]) for s in sids])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(sids), len(sids))
        boots.append(sums[idx].sum() / cnts[idx].sum())
    return {"mean": float(sums.sum() / cnts.sum()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--scorer", default=None, help="Stage-1 scorer.pt; without it arm B is skipped.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budgets", default="8,16,32,64")
    ap.add_argument("--max_streams", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    v, dcfg, tcfg = cfg["vismem"], cfg["data"], cfg["training"]
    fps, rho, tol, seed = float(dcfg["sample_fps"]), float(v["lvm_saw_ema_decay"]), float(tcfg["time_tolerance"]), int(tcfg["seed"])
    P = {"l_min": int(v["lvm_saw_l_min"]), "l_max": int(v["lvm_saw_l_max"]), "quantile": float(v["lvm_saw_quantile"]),
         "window": int(v["lvm_saw_recent_window"]), "b_s": float(v["lvm_saw_energy_budget"])}
    n_main = int(v["lvm_num_slots"])
    os.makedirs(args.out_dir, exist_ok=True)
    cache = FrameCache(args.feature_cache)
    scorer = None
    if args.scorer:
        st = torch.load(args.scorer, map_location="cpu", weights_only=False)
        scorer = ImportanceScorer(3584, proj_dim=int(st.get("proj_dim", v["lvm_scorer_proj_dim"])))
        scorer.load_state_dict(st["scorer"])
        scorer.eval()
    arms = ["A", "B", "C", "D"] if scorer is not None else ["A", "C", "D"]

    def ds(split, source, part):
        d = Ego4DStreamDataset(args.clean_dir, "", split, source, part=part, dev_frac=float(dcfg["dev_frac"]), sample_fps=fps, seed=seed,
                               require_video=False, query_time_mode=str(dcfg.get("query_time_mode", "uniform")))
        d.items = [i for i in d.items if all(cache.has(u) for u in i.clip_uids)][: args.max_streams]
        d.items = [d[i] for i in range(len(d.items))]  # attach the fixed (epoch-0) ask times
        return d

    t0 = time.time()
    seen = train_participants(args.clean_dir)
    train = load_streams(ds("train", "merged", "train"), cache, fps, rho, tol, scorer, None)
    print(f"[replay] train-fit streams {len(train)} loaded ({time.time() - t0:.0f}s)", flush=True)

    # Calibration on train-fit: arm A's cut count is the target for every arm.
    cal = {}
    cuts_a = [cut_points(s["A"], s["cut"], P["l_min"], P["l_max"], P["b_s"], P["quantile"], P["window"]) for s in train]
    target = float(np.mean([len(c) for c in cuts_a]))
    mean_len = float(np.mean([len(s["t"]) / max(len(c), 1) for s, c in zip(train, cuts_a)]))
    cal["A"] = {"b_s": P["b_s"], "quantile": P["quantile"], "segments": target, "tau_s": float(np.median([m for s, c in zip(train, cuts_a) for m in seg_means(s["A"], c)]))}
    for arm in [a for a in arms if a in ("B", "C")]:
        b_s, q, got, note = match_budget(train, arm, target, P)
        cuts = [cut_points(s[arm], s["cut"], P["l_min"], P["l_max"], b_s, q, P["window"]) for s in train]
        cal[arm] = {"b_s": b_s, "quantile": q, "segments": got, "note": note, "tau_s": float(np.median([m for s, c in zip(train, cuts) for m in seg_means(s[arm], c)]))}
    L = int(round(mean_len))
    cuts_d = [cut_points(s["D"], s["cut"], 0, 0, 0, 0, 0, fixed_len=L) for s in train]
    # Online, arm D writes w = 0 and keeps the config tau_s > 0, so its update rule is similarity alone.
    cal["D"] = {"fixed_len": L, "segments": float(np.mean([len(c) for c in cuts_d])), "tau_s": float("inf")}
    # tau_r from arm A's own writes at the main budget (nearest-node cosine, 0.9 quantile).
    near = []
    for s, c in zip(train, cuts_a):
        _, mem = replay_stream(s, "A", c, n_main, tau_r=2.0, tau_s=-1.0, queries=[])
        near += mem.near_cos
    tau_r = float(np.quantile(near, 0.9))
    for arm in cal:
        cal[arm]["tau_r"] = tau_r
    print(f"[replay] calibration ({time.time() - t0:.0f}s): {json.dumps(cal)}", flush=True)
    del train

    results = {"calibration": cal, "tracks": {}}
    for track, src in (("nlq", "nlq"), ("qa", "qaego4d")):
        val = load_streams(ds("val", src, "all"), cache, fps, rho, tol, scorer, seen)
        per = {}
        for N in [int(x) for x in args.budgets.split(",")]:
            for arm in arms:
                c = cal[arm]
                rows = []
                for s in val:
                    if arm == "D":
                        cuts = cut_points(s["D"], s["cut"], 0, 0, 0, 0, 0, fixed_len=c["fixed_len"])
                    else:
                        cuts = cut_points(s[arm], s["cut"], P["l_min"], P["l_max"], c["b_s"], c["quantile"], P["window"])
                    r, _ = replay_stream(s, arm, cuts, N, c["tau_r"], c["tau_s"], s["item"].queries)
                    for x in r:
                        x["segments"] = len(cuts)
                    rows += r
                per[(N, arm)] = rows
        out = {}
        for (N, arm), rows in per.items():
            ent = {k: float(np.mean([r[k] for r in rows])) for k in ("survival", "best_iou", "survival_committed", "best_iou_committed", "num_nodes", "mean_span_s")}
            ent["queries"] = len(rows)
            ent["by_delay"] = {b: float(np.mean([r["best_iou"] for r in rows if r["delay_bucket"] == b]))
                               for b in sorted({r["delay_bucket"] for r in rows})}
            ent["disjoint_best_iou"] = float(np.mean([r["best_iou"] for r in rows if r["disjoint"]])) if any(r["disjoint"] for r in rows) else None
            if arm != "A":
                ent["vs_A"] = {k: bootstrap_diff(per[(N, "A")], rows, k) for k in ("best_iou", "survival", "best_iou_committed")}
            out[f"N={N}/{arm}"] = ent
            print(f"[replay] {track} N={N} {arm}: IoU*={ent['best_iou']:.4f} surv={ent['survival']:.4f} "
                  f"IoU*_committed={ent['best_iou_committed']:.4f} nodes={ent['num_nodes']:.1f} span={ent['mean_span_s']:.0f}s "
                  f"{'vs A ' + json.dumps(ent['vs_A']['best_iou']) if arm != 'A' else ''}", flush=True)
        results["tracks"][track] = out
    with open(os.path.join(args.out_dir, "check2.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
