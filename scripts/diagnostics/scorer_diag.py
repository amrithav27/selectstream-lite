"""Does the Stage-1 scorer learn anything the write rule can use?

The cut rule compares w_t with a quantile of the last 64 frames of the same stream, so only
within-stream (local) ranking matters. Pooled AP/AUC also rewards knowing which streams have many
asked-about frames, which the write rule cannot use. Reported per score:
  pooled AUC/AP, offset-only AUC (each frame scored by its stream's mean score),
  within-stream AUC (frame-weighted mean over streams with both classes, bootstrap CI over streams),
  local AUC (score minus its causal 64-frame trailing mean, pooled),
  null within-stream AUC with labels circularly shifted inside each stream.
Scorers: the saved scorer.pt (arm B), and retrained on CPU with equal budgets:
  full, scalars-only [s, sbar], g-only, full trained on circularly shifted labels (keeps each stream's
  positive rate and label autocorrelation, breaks alignment with the frames).
"""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from streamlite.cli.common import load_yaml
from streamlite.cli.ego4d_common import train_participants
from streamlite.cli.train_scorer import auroc, average_precision, ema, load_split
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache
from streamlite.model.importance_scorer import ImportanceScorer, scorer_bce

CACHE, SCORER, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
torch.set_num_threads(6)
rng = np.random.default_rng(0)
cfg = load_yaml("configs/ego4d_qwen25vl7b.yaml")
v, d = cfg["vismem"], cfg["data"]
fps, rho, tol, seed0 = float(d["sample_fps"]), float(v["lvm_saw_ema_decay"]), float(cfg["training"]["time_tolerance"]), int(cfg["training"]["seed"])
WIN = int(v["lvm_saw_recent_window"])
cache = FrameCache(CACHE)
seen = train_participants("dataset/clean")


def ds(split, source, part):
    x = Ego4DStreamDataset("dataset/clean", "", split, source, part=part, dev_frac=float(d["dev_frac"]), sample_fps=fps, seed=seed0, require_video=False)
    x.items = [i for i in x.items if all(cache.has(u) for u in i.clip_uids)]
    return x


t0 = time.time()
train = load_split(ds("train", "merged", "train"), cache, fps, rho, tol, "cpu")
dev = load_split(ds("train", "merged", "dev"), cache, fps, rho, tol, "cpu")
val = {tr: load_split(ds("val", src, "all"), cache, fps, rho, tol, "cpu", seen) for tr, src in (("nlq", "nlq"), ("qa", "qaego4d"))}
print(f"loaded {len(train)} train, {len(dev)} dev, {len(val['nlq'])}/{len(val['qa'])} val streams in {time.time() - t0:.0f}s", flush=True)


def shift(y):
    return np.roll(y, int(rng.integers(len(y) // 4, max(len(y) // 4 + 1, 3 * len(y) // 4)))) if len(y) > 4 else y


for st in train + dev:
    st["y_shift"] = shift(st["y"])


class Masked(ImportanceScorer):
    def __init__(self, D, mode):
        super().__init__(D, proj_dim=16)
        self.mode = mode

    def forward_batch(self, sc, g):
        if self.mode == "scalars":
            g = torch.zeros_like(g)
        elif self.mode == "g":
            sc = torch.zeros_like(sc)
        return super().forward_batch(sc, g)


def scores(model, streams):
    with torch.no_grad():
        for st in streams:
            st["_a"] = torch.sigmoid(model.forward_batch(torch.from_numpy(st["sc"]), st["g"].float())).numpy().astype(np.float64)
    return [st["_a"] for st in streams]


def fit(mode, label_key):
    torch.manual_seed(seed0)
    m = Masked(train[0]["g"].shape[1], mode)
    G = torch.cat([s["g"] for s in train])
    S = torch.from_numpy(np.concatenate([s["sc"] for s in train]))
    Y = torch.from_numpy(np.concatenate([s[label_key] for s in train]).astype(np.float32))
    pw = float((len(Y) - Y.sum()) / Y.sum())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed0)
    best, best_state, log = -1, None, []
    ydev = np.concatenate([s[label_key] for s in dev])
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Y), generator=gen)
        for i in range(0, len(Y), 4096):
            idx = perm[i : i + 4096]
            loss = scorer_bce(m.forward_batch(S[idx], G[idx].float()), Y[idx], pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ap = average_precision(np.concatenate(scores(m, dev)), ydev)
        log.append(round(ap, 4))
        if ap > best:
            best, best_state = ap, {k: x.clone() for k, x in m.state_dict().items()}
    m.load_state_dict(best_state)
    print(f"[fit] {mode}/{label_key}: dev AP by epoch {log}", flush=True)
    return m


def boot_ci(vals, w, n=1000):
    vals, w = np.asarray(vals), np.asarray(w, float)
    idx = rng.integers(0, len(vals), (n, len(vals)))
    m = (vals[idx] * w[idx]).sum(1) / w[idx].sum(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def analyse(per_stream_scores, streams):
    y = np.concatenate([s["y"] for s in streams])
    x = np.concatenate(per_stream_scores)
    off = np.concatenate([np.full(len(a), a.mean()) for a in per_stream_scores])
    loc = np.concatenate([a - np.convolve(np.concatenate([np.full(WIN - 1, a[0]), a]), np.ones(WIN) / WIN, "valid") for a in per_stream_scores])
    within, wts, null = [], [], []
    for a, s in zip(per_stream_scores, streams):
        if 0 < s["y"].sum() < len(s["y"]):
            within.append(auroc(a, s["y"]))
            wts.append(len(a))
            null.append(np.mean([auroc(a, shift(s["y"])) for _ in range(5)]))
    wm = float(np.average(within, weights=wts))
    return {"pooled_auc": auroc(x, y), "pooled_ap": average_precision(x, y), "base": float(y.mean()), "offset_only_auc": auroc(off, y),
            "within_auc": wm, "within_ci": boot_ci(within, wts), "within_null": float(np.average(null, weights=wts)),
            "local_auc": auroc(loc, y), "streams_used": len(within)}


sets = {"dev": dev, "val_nlq": val["nlq"], "val_qa": val["qa"], "val_nlq_disj": [s for s in val["nlq"] if s["disjoint"]]}
res = {}
saved = ImportanceScorer(train[0]["g"].shape[1], proj_dim=16)
saved.load_state_dict(torch.load(SCORER, map_location="cpu", weights_only=False)["scorer"])
models = {"saved(armB)": saved}
for mode, lab in (("full", "y"), ("scalars", "y"), ("g", "y"), ("full", "y_shift")):
    models[f"{mode}{'/shiftedY' if lab != 'y' else ''}"] = fit(mode, lab)
for name, streams in sets.items():
    res[name] = {"sbar": analyse([s["sc"][:, 1].astype(np.float64) for s in streams], streams)}
    for mname, m in models.items():
        a = scores(m, streams)
        res[name][mname] = analyse(a, streams)
        if mname == "saved(armB)":
            res[name]["w=EMA(a)"] = analyse([ema(x, rho) for x in a], streams)
    for k, r in res[name].items():
        print(f"{name:13s} {k:16s} pooledAUC {r['pooled_auc']:.3f} AP {r['pooled_ap']:.3f} (base {r['base']:.3f}) | offset-only {r['offset_only_auc']:.3f} | "
              f"within {r['within_auc']:.3f} [{r['within_ci'][0]:.3f},{r['within_ci'][1]:.3f}] null {r['within_null']:.3f} | local {r['local_auc']:.3f} | n={r['streams_used']}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
