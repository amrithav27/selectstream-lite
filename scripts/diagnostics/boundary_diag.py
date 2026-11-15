"""(1) Is the scorer's within-video signal uniform across videos, or concentrated? Paired per-video
AUC of the scorer vs surprise, by scenario, template and seen/unseen participant.
(2) Do asked-about frames sit at event boundaries? Boundary detectors on the cached g_t (no new model):
  noncausal b_k(t) = 1 - cos(mean g[t-k:t], mean g[t:t+k])   (GEBD-style; an upper bound for any
                    boundary detector on these features, since it sees k future frames)
  causal    c_k(t) = 1 - cos(mean g[t-2k:t-k], mean g[t-k:t]) (deployable online, k frames late)
  near-peak: frames within d s of a detected boundary (local max of b_8 above the stream's 80th pct).
(3) Answer-window length vs segment length: what IoU is possible at this cut budget.
Val NLQ track, all 300 streams.
"""
import collections
import json
import sys

import numpy as np
import torch

from streamlite.cli.common import load_yaml
from streamlite.cli.ego4d_common import train_participants
from streamlite.cli.train_scorer import auroc, ema, load_split
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache
from streamlite.model.importance_scorer import ImportanceScorer

CACHE, SCORER, OUT = sys.argv[1:4]
torch.set_num_threads(6)
rng = np.random.default_rng(0)
cfg = load_yaml("configs/ego4d_qwen25vl7b.yaml")
v, d = cfg["vismem"], cfg["data"]
fps, rho, tol, seed0 = float(d["sample_fps"]), float(v["lvm_saw_ema_decay"]), float(cfg["training"]["time_tolerance"]), int(cfg["training"]["seed"])
cache = FrameCache(CACHE)
seen = train_participants("dataset/clean")
dsv = Ego4DStreamDataset("dataset/clean", "", "val", "nlq", part="all", dev_frac=float(d["dev_frac"]), sample_fps=fps, seed=seed0, require_video=False)
dsv.items = [i for i in dsv.items if all(cache.has(u) for u in i.clip_uids)]
items = {i.stream_id: i for i in dsv.items}
val = load_split(dsv, cache, fps, rho, tol, "cpu", seen)

meta = json.load(open("dataset/metadata/ego4d.json"))
scen = {}
for vid in meta.get("videos", []):
    s = vid.get("scenarios") or []
    scen[vid.get("video_uid")] = s[0] if s else "unknown"

sc = ImportanceScorer(val[0]["g"].shape[1], proj_dim=16)
sc.load_state_dict(torch.load(SCORER, map_location="cpu", weights_only=False)["scorer"])


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def boundary(g, k, causal):
    T = len(g)
    c = np.concatenate([np.zeros((1, g.shape[1])), np.cumsum(g, 0)])
    out = np.zeros(T)
    for t in range(T):
        if causal:
            a0, a1, b0, b1 = t - 2 * k + 1, t - k + 1, t - k + 1, t + 1
        else:
            a0, a1, b0, b1 = t - k, t, t, t + k
        a0, b1 = max(a0, 0), min(b1, T)
        if a1 <= a0 or b1 <= b0:
            continue
        ma, mb = (c[a1] - c[a0]) / (a1 - a0), (c[b1] - c[b0]) / (b1 - b0)
        out[t] = 1 - float(unit(ma) @ unit(mb))
    return out


def peaks(b, pct=80, nms=4):
    thr = np.percentile(b, pct)
    p = [t for t in range(len(b)) if b[t] >= thr and b[t] == b[max(0, t - nms):t + nms + 1].max()]
    return np.array(p, dtype=int)


rows = []
for st in val:
    g = st["g"].float().numpy()
    gu = unit(g)
    with torch.no_grad():
        a = torch.sigmoid(sc.forward_batch(torch.from_numpy(st["sc"]), st["g"].float())).numpy().astype(np.float64)
    y = st["y"]
    S = {"sbar": st["sc"][:, 1].astype(np.float64), "s": st["sc"][:, 0].astype(np.float64), "a": a, "w": ema(a, rho)}
    for k in (2, 4, 8, 16):
        S[f"b{k}"] = boundary(gu, k, False)
        S[f"c{k}"] = boundary(gu, k, True)
    pk = peaks(S["b8"])
    dist = np.full(len(y), 1e9)
    for p in pk:
        dist = np.minimum(dist, np.abs(np.arange(len(y)) - p))
    S["near_b8"] = -dist
    item = items[st["id"]]
    wl = [q.window_end - q.window_start for q in item.queries]
    r = {"id": st["id"], "T": len(y), "pos": float(y.mean()), "nq": len(item.queries), "disjoint": st["disjoint"], "scenario": scen.get(st["id"], "unknown"),
         "win_len": wl, "templates": [q.template for q in item.queries], "n_peaks": len(pk),
         "enrich": {dd: float(y[dist <= dd].mean() / max(y.mean(), 1e-9)) if (dist <= dd).any() else None for dd in (0, 2, 5, 10)}}
    if 0 < y.sum() < len(y):
        r["auc"] = {k: auroc(x, y) for k, x in S.items()}
    rows.append(r)
print(f"{len(rows)} streams, {sum('auc' in r for r in rows)} with both classes", flush=True)

R = [r for r in rows if "auc" in r]
W = np.array([r["T"] for r in R], float)


def ci(x, w):
    idx = rng.integers(0, len(x), (2000, len(x)))
    m = (x[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return float(np.average(x, weights=w)), float(np.quantile(m, .025)), float(np.quantile(m, .975))


res = {"signals": {}, "paired": {}, "by_scenario": {}, "windows": {}, "enrichment": {}}
print("\n== within-video AUC, frame-weighted mean [95% CI over videos] ==")
for k in R[0]["auc"]:
    x = np.array([r["auc"][k] for r in R])
    m, lo, hi = ci(x, W)
    res["signals"][k] = [m, lo, hi]
    print(f"{k:8s} {m:.3f} [{lo:.3f},{hi:.3f}]")

print("\n== per video: scorer a vs surprise sbar ==")
diff = np.array([r["auc"]["a"] - r["auc"]["sbar"] for r in R])
m, lo, hi = ci(diff, W)
q = np.quantile(diff, [.1, .25, .5, .75, .9])
res["paired"] = {"mean": [m, lo, hi], "frac_a_better": float((diff > 0).mean()), "quantiles": q.tolist()}
print(f"a - sbar: mean {m:+.3f} [{lo:+.3f},{hi:+.3f}]; a better on {(diff > 0).mean():.0%} of {len(R)} videos; deciles 10/25/50/75/90: {np.round(q, 3)}")
aa = np.array([r["auc"]["a"] for r in R])
print(f"scorer per-video AUC: deciles {np.round(np.quantile(aa, [.1, .25, .5, .75, .9]), 3)}; >0.6 on {(aa > .6).mean():.0%}, <0.4 on {(aa < .4).mean():.0%}")
for name, f in (("seen participant", lambda r: not r["disjoint"]), ("unseen participant", lambda r: r["disjoint"])):
    sel = [i for i, r in enumerate(R) if f(r)]
    if len(sel) > 5:
        mm = ci(diff[sel], W[sel])
        print(f"  {name:18s} n={len(sel):3d} a-sbar {mm[0]:+.3f} [{mm[1]:+.3f},{mm[2]:+.3f}]  a {np.average(aa[sel], weights=W[sel]):.3f}")
        res["paired"][name] = {"n": len(sel), "diff": mm}

print("\n== by scenario (>= 8 videos) ==")
groups = collections.defaultdict(list)
for i, r in enumerate(R):
    groups[r["scenario"]].append(i)
for sname, sel in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    if len(sel) < 8:
        continue
    out = {k: float(np.average([R[i]["auc"][k] for i in sel], weights=W[sel])) for k in ("sbar", "a", "w", "b8", "c4")}
    res["by_scenario"][sname] = {"n": len(sel), **out}
    print(f"  {sname[:38]:38s} n={len(sel):3d} " + " ".join(f"{k} {x:.3f}" for k, x in out.items()))

print("\n== answer windows vs segments ==")
wl = np.array([x for r in rows for x in r["win_len"]])
seg = np.array([r["T"] for r in rows]) / 28.2  # arm A: 28.2 segments per stream (calibration)
res["windows"] = {"win_len_q": np.quantile(wl, [.1, .25, .5, .75, .9]).tolist(), "seg_len_median": float(np.median(seg))}
print(f"window length s, deciles 10/25/50/75/90: {np.round(np.quantile(wl, [.1, .25, .5, .75, .9]), 1)}; mean segment at arm A's budget: median {np.median(seg):.1f} s")
print(f"best possible IoU of one segment of median length with a median window: {min(np.median(wl), np.median(seg)) / max(np.median(wl), np.median(seg)):.2f}")
print(f"windows shorter than L_min = 8 s: {(wl < 8).mean():.0%}")

print("\n== positive rate near detected boundaries (b8 peaks, top 20%, NMS 4 frames) / base rate ==")
for dd in (0, 2, 5, 10):
    e = [r["enrich"][dd] for r in rows if r["enrich"][dd] is not None]
    res["enrichment"][dd] = float(np.mean(e))
    print(f"  within {dd:2d} s of a boundary: x{np.mean(e):.2f}")
json.dump({"summary": res, "rows": rows}, open(OUT, "w"))
