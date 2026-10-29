"""Stage 1 (docs/plan.md §4) and Check 1: train the Q1 scorer on cached g_t; does anything predict
asked-about moments?

The scorer's loss never depended on the graph, so it is trained here on every train-fit frame instead of
inside Stage R. Inputs are recomputed from the cache exactly as the online write path builds them:
s_t = 1 - cos(g_t, g_{t-1}) (0 on the first frame), sbar_t = EMA_rho(s_t) from 0, phi_t = [s_t, sbar_t, g_t].
Label y_t = 1[t within tol of any answer window of the stream's queries] (frame_labels).

Reported (AP and AUROC of each score against y_t, with the base rate) on dev, val (NLQ and QA tracks)
and the participant-disjoint part of val:
  s_t, sbar_t          surprise (arm A's signal is sbar_t)
  a_t, EMA(a_t)        the scorer; EMA(a_t) is arm B's write signal w_t
Variants: proj_dim 16 (the model's, --seeds seeds; the best by the selection metric is saved for arm B) and
larger proj_dim (one seed, diagnostic: does the 16-d bottleneck limit the signal, or does a wider one only
help on participants seen in training?).

Pooled AP/AUROC also reward knowing which streams have many asked-about frames; the cut rule compares w_t
with the last 64 frames of the same stream and cannot use that. So every score also gets (docs/plan.md §13.1):
  within_auc   frame-weighted mean over streams of the AUROC inside each stream
  local_auc    AUROC of the score minus its causal 64-frame trailing mean, pooled
  offset_auc   AUROC when every frame gets its stream's mean score (stream identity alone)
Ablations (docs/plan.md §13.3): --input_mask (AB-I1), --loss (AB-L2), --time_tolerance (AB-L1) and
--label_shift (control: labels circularly shifted inside each stream keep each stream's positive rate and
autocorrelation but not their alignment with the frames).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from main.cli.common import load_yaml
from main.cli.ego4d_common import frame_labels, log_device, train_participants
from main.data.ego4d_qa import Ego4DStreamDataset
from main.data.frame_cache import FrameCache
from main.model.importance_scorer import INPUT_MASKS, ImportanceScorer, scorer_bce

LOSSES = ("bce", "stream_bce", "rank_local")
LOCAL_WINDOW = 64  # = lvm_saw_recent_window; overwritten from the config in main()


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    y = y[order]
    if y.sum() == 0:
        return float("nan")
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, n_pos = y == 1, int((y == 1).sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ema(x: np.ndarray, rho: float) -> np.ndarray:
    out, prev = np.empty_like(x), 0.0
    for i, v in enumerate(x):
        prev = rho * prev + (1.0 - rho) * v
        out[i] = prev
    return out


def surprise(g: torch.Tensor, rho: float):
    """s_t, sbar_t as LVMMemoryFormer._compute_chunk_surprise with the attention term off."""
    s = torch.zeros(g.shape[0], dtype=torch.float32, device=g.device)
    if g.shape[0] > 1:
        s[1:] = 1.0 - F.cosine_similarity(g[1:], g[:-1], dim=-1)
    s = s.cpu().numpy().astype(np.float64)
    sbar = ema(s, rho)
    sbar[0] = 0.0
    return s, sbar


def load_split(ds, cache, fps, rho, tol, device, seen=None):
    """Per stream: g (fp16, cpu), scalars [T,2], y [T], stream id, disjoint flag."""
    out = []
    for item in ds.items:
        g, t, _ = cache.stream_g(item, fps)
        if len(t) == 0:
            continue
        gt = torch.from_numpy(g).to(device)
        s, sbar = surprise(gt, rho)
        y = frame_labels(t, [(q.window_start, q.window_end) for q in item.queries], tol)
        out.append({"id": item.stream_id, "g": gt.half().cpu(), "sc": np.stack([s, sbar], 1).astype(np.float32), "y": y.astype(np.float32),
                    "disjoint": seen is not None and item.participant_id is not None and item.participant_id not in seen})
    return out


@torch.no_grad()
def predict(scorer, streams, device, rho):
    for st in streams:
        logits = []
        for i in range(0, len(st["y"]), 8192):
            g = st["g"][i : i + 8192].to(device).float()
            sc = torch.from_numpy(st["sc"][i : i + 8192]).to(device)
            logits.append(scorer.forward_batch(sc, g).float().cpu())
        a = torch.sigmoid(torch.cat(logits)).numpy().astype(np.float64)
        st["a"], st["w"] = a, ema(a, rho)


def stream_score(st, k):
    if k in ("s", "sbar"):
        return st["sc"][:, 0 if k == "s" else 1].astype(np.float64)
    return st[k]


def within_auc(xs, ys):
    """Frame-weighted mean of the per-stream AUROC over streams with both classes."""
    vals, wts = [], []
    for x, y in zip(xs, ys):
        if 0 < y.sum() < len(y):
            vals.append(auroc(x, y))
            wts.append(len(y))
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def trailing_mean(x, window):
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(i - window + 1, 0)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def metrics(streams, keys=("s", "sbar", "a", "w")):
    if not streams:
        return None
    ys = [s["y"] for s in streams]
    y = np.concatenate(ys)
    res = {"frames": int(len(y)), "streams": len(streams), "base_rate": float(y.mean())}
    for k in keys:
        if k not in ("s", "sbar") and k not in streams[0]:
            continue
        xs = [stream_score(s, k) for s in streams]
        x = np.concatenate(xs)
        res[k] = {"ap": average_precision(x, y), "auroc": auroc(x, y), "within_auc": within_auc(xs, ys),
                  "local_auc": auroc(np.concatenate([a - trailing_mean(a, LOCAL_WINDOW) for a in xs]), y),
                  "offset_auc": auroc(np.concatenate([np.full(len(a), a.mean()) for a in xs]), y)}
    return res


def frame_weights(train, loss):
    """Per-frame BCE weights. bce: none (pos_weight does the class balance). stream_bce: each stream's
    positives and negatives get half its weight each, so a per-stream constant earns nothing and only
    within-stream ranking lowers the loss; streams without positives (or negatives) are dropped."""
    if loss != "stream_bce":
        return None
    out = []
    for s in train:
        y, n = s["y"], len(s["y"])
        p = float(y.sum())
        if p == 0 or p == n:
            out.append(np.zeros(n, np.float32))
            continue
        out.append(np.where(y > 0, 0.5 * n / p, 0.5 * n / (n - p)).astype(np.float32))
    w = np.concatenate(out)
    return w / w[w > 0].mean()


def local_pairs(train, n, window, gen):
    """n (positive, negative) global frame indices: the negative from the `window` frames before the positive
    in the same stream, the comparison the cut rule makes (w_t vs the last 64 frames)."""
    pos, neg_lo, neg_hi, starts = [], [], [], 0
    for s in train:
        y = s["y"]
        pi = np.flatnonzero(y > 0)
        if len(pi) and (y == 0).any():
            pos.append(pi + starts)
            neg_lo.append(np.maximum(pi - window, 0) + starts)
            neg_hi.append(np.maximum(pi - 1, 0) + starts)  # draws that land on a positive are dropped below
        starts += len(y)
    pos, lo, hi = np.concatenate(pos), np.concatenate(neg_lo), np.concatenate(neg_hi)
    Y = np.concatenate([s["y"] for s in train])
    rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=gen)))
    k = rng.integers(0, len(pos), n * 3)
    j = lo[k] + (rng.random(len(k)) * (hi[k] - lo[k] + 1)).astype(np.int64)
    ok = Y[j] == 0  # windows with no negative are rare; drop those draws
    return pos[k][ok][:n], j[ok][:n]


def train_one(train, dev, proj_dim, seed, pos_weight, device, rho, args):
    torch.manual_seed(seed)
    scorer = ImportanceScorer(train[0]["g"].shape[1], proj_dim=proj_dim, input_mask=args.input_mask).to(device)
    # forward_batch must be the same function as the per-frame forward the online path calls.
    with torch.no_grad():
        g0, s0 = train[0]["g"][:4].to(device).float(), torch.from_numpy(train[0]["sc"][:4]).to(device)
        ref = torch.stack([scorer(s0[i], g0[i]) for i in range(4)])
        assert torch.allclose(ref, scorer.forward_batch(s0, g0), atol=1e-5), "forward_batch != forward"
    G = torch.cat([s["g"] for s in train]).to(device)  # fp16 on GPU
    S = torch.from_numpy(np.concatenate([s["sc"] for s in train])).to(device)
    Y = torch.from_numpy(np.concatenate([s["y"] for s in train])).to(device)
    fw = frame_weights(train, args.loss)
    FW = torch.from_numpy(fw).to(device) if fw is not None else None
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad, log = -1.0, None, 0, []
    for epoch in range(args.max_epochs):
        scorer.train()
        tot = 0.0
        if args.loss == "rank_local":
            # As many pairs per epoch as there are frames; logistic loss on (positive - nearby negative).
            pi, ni = (torch.from_numpy(x).to(device) for x in local_pairs(train, len(Y), LOCAL_WINDOW, gen))
            for i in range(0, len(pi), args.batch_size):
                p, n = pi[i : i + args.batch_size], ni[i : i + args.batch_size]
                lp = scorer.forward_batch(S[p], G[p].float())
                ln = scorer.forward_batch(S[n], G[n].float())
                loss = F.softplus(-(lp - ln)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(p)
            n_seen = len(pi)
        else:
            perm = torch.randperm(len(Y), generator=gen).to(device)
            for i in range(0, len(Y), args.batch_size):
                idx = perm[i : i + args.batch_size]
                logits = scorer.forward_batch(S[idx], G[idx].float())
                if FW is None:
                    loss = scorer_bce(logits, Y[idx], pos_weight)
                else:
                    loss = (F.binary_cross_entropy_with_logits(logits.float(), Y[idx], reduction="none") * FW[idx]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            n_seen = len(Y)
        scorer.eval()
        predict(scorer, dev, device, rho)
        m = metrics(dev, ("a",))["a"]
        dev_sel = m["ap"] if args.select == "ap" else m["within_auc"]
        log.append({"epoch": epoch, "train_loss": tot / n_seen, "dev_ap_a": m["ap"], "dev_within_auc_a": m["within_auc"]})
        print(f"[scorer] proj={proj_dim} seed={seed} epoch={epoch} {args.loss}={tot / n_seen:.4f} dev_AP(a)={m['ap']:.4f} "
              f"dev_within_AUC(a)={m['within_auc']:.4f}", flush=True)
        if dev_sel > best + 1e-4:
            best, bad = dev_sel, 0
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    scorer.load_state_dict(best_state)
    del G, S, Y
    torch.cuda.empty_cache()
    return scorer, best, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--diag_proj_dims", default="128", help="Comma list of extra proj_dims, one seed each (diagnostic).")
    ap.add_argument("--max_streams", type=int, default=None, help="Pilot: first N streams of every split.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--input_mask", choices=sorted(INPUT_MASKS), default="all", help="AB-I1: which inputs the scorer sees.")
    ap.add_argument("--loss", choices=LOSSES, default="bce", help="AB-L2: bce = pooled weighted BCE (Stage 1); stream_bce = per-stream "
                    "balanced BCE; rank_local = positive vs a negative <= 64 frames away in the same stream.")
    ap.add_argument("--select", choices=("ap", "within_auc"), default=None, help="Dev metric for early stopping and seed choice "
                    "(default: ap for bce, within_auc otherwise).")
    ap.add_argument("--time_tolerance", type=float, default=None, help="AB-L1: label widening in s (default: config training.time_tolerance).")
    ap.add_argument("--label_shift", action="store_true", help="Control: train and select on labels circularly shifted inside each stream.")
    args = ap.parse_args()
    args.select = args.select or ("ap" if args.loss == "bce" else "within_auc")

    cfg = load_yaml(args.config)
    v, dcfg, tcfg = cfg["vismem"], cfg["data"], cfg["training"]
    fps, rho, tol, seed0 = float(dcfg["sample_fps"]), float(v["lvm_saw_ema_decay"]), float(tcfg["time_tolerance"]), int(tcfg["seed"])
    if args.time_tolerance is not None:
        tol = args.time_tolerance
    global LOCAL_WINDOW
    LOCAL_WINDOW = int(v["lvm_saw_recent_window"])
    proj_dim = int(v["lvm_scorer_proj_dim"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    log_device()
    cache = FrameCache(args.feature_cache)
    seen = train_participants(args.clean_dir)

    def ds(split, source, part):
        d = Ego4DStreamDataset(args.clean_dir, "", split, source, part=part, dev_frac=float(dcfg["dev_frac"]), sample_fps=fps, seed=seed0, require_video=False)
        n0 = len(d.items)
        d.items = [i for i in d.items if all(cache.has(u) for u in i.clip_uids)][: args.max_streams]
        print(f"[data] {split}/{source}/{part}: {len(d.items)} of {n0} streams cached", flush=True)
        return d

    t0 = time.time()
    train = load_split(ds("train", "merged", "train"), cache, fps, rho, tol, device)
    dev = load_split(ds("train", "merged", "dev"), cache, fps, rho, tol, device)
    val = {tr: load_split(ds("val", src, "all"), cache, fps, rho, tol, device, seen) for tr, src in (("nlq", "nlq"), ("qa", "qaego4d"))}
    if args.label_shift:
        rng = np.random.default_rng(seed0)
        for s in train + dev:
            n = len(s["y"])
            s["y"] = np.roll(s["y"], int(rng.integers(n // 4, 3 * n // 4 + 1))) if n > 4 else s["y"]
    y_tr = np.concatenate([s["y"] for s in train])
    pos_weight = float((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0))
    print(f"[data] train frames={len(y_tr)} pos_rate={y_tr.mean():.4f} pos_weight={pos_weight:.3f}; loaded in {time.time() - t0:.0f}s", flush=True)

    def report(scorer):
        out = {}
        for name, streams in (("dev", dev), ("val_nlq", val["nlq"]), ("val_qa", val["qa"])):
            predict(scorer, streams, device, rho)
            out[name] = metrics(streams)
        for tr in ("nlq", "qa"):
            out[f"val_{tr}_disjoint"] = metrics([s for s in val[tr] if s["disjoint"]])
        return out

    results = {"pos_weight": pos_weight, "train_frames": int(len(y_tr)), "train_pos_rate": float(y_tr.mean()), "runs": [],
               "ablation": {"input_mask": args.input_mask, "loss": args.loss, "select": args.select, "time_tolerance": tol, "label_shift": args.label_shift}}
    best = None
    for k in range(args.seeds):
        scorer, dev_ap, log = train_one(train, dev, proj_dim, seed0 + k, pos_weight, device, rho, args)
        rep = report(scorer)
        results["runs"].append({"proj_dim": proj_dim, "seed": seed0 + k, "dev_ap": dev_ap, "log": log, "metrics": rep})
        print(json.dumps({"proj_dim": proj_dim, "seed": seed0 + k, "metrics": rep}), flush=True)
        if best is None or dev_ap > best[1]:  # dev_ap here is the selection metric (args.select)
            best = (scorer, dev_ap, seed0 + k, rep)
    for pd in [int(x) for x in args.diag_proj_dims.split(",") if x]:
        scorer, dev_ap, log = train_one(train, dev, pd, seed0, pos_weight, device, rho, args)
        rep = report(scorer)
        results["runs"].append({"proj_dim": pd, "seed": seed0, "dev_ap": dev_ap, "log": log, "metrics": rep, "diagnostic": True})
        print(json.dumps({"proj_dim": pd, "seed": seed0, "metrics": rep}), flush=True)

    scorer, dev_ap, seed, rep = best
    torch.save({"scorer": {k: v.cpu() for k, v in scorer.state_dict().items()}, "proj_dim": proj_dim, "seed": seed, "dev_ap": dev_ap,
                "pos_weight": pos_weight, "metrics": rep, "input_mask": args.input_mask, **results["ablation"]},
               os.path.join(args.out_dir, "scorer.pt"))
    ap16 = [r["metrics"]["val_nlq"]["a"]["ap"] for r in results["runs"] if r["proj_dim"] == proj_dim]
    results["selected"] = {"seed": seed, "dev_ap": dev_ap, "val_nlq_ap_a_over_seeds": {"mean": float(np.mean(ap16)), "std": float(np.std(ap16))}}
    with open(os.path.join(args.out_dir, "check1.json"), "w") as f:
        json.dump(results, f, indent=1)

    predict(scorer, val["nlq"], device, rho)
    vn = val["nlq"]
    ex = max(vn, key=lambda st: len(st["y"]))
    np.savez_compressed(
        os.path.join(args.out_dir, "check1_scores.npz"),
        val_nlq_y=np.concatenate([st["y"] for st in vn]).astype(np.int8),
        val_nlq_s=np.concatenate([st["sc"][:, 0] for st in vn]).astype(np.float32),
        val_nlq_sbar=np.concatenate([st["sc"][:, 1] for st in vn]).astype(np.float32),
        val_nlq_a=np.concatenate([st["a"] for st in vn]).astype(np.float32),
        val_nlq_w=np.concatenate([st["w"] for st in vn]).astype(np.float32),
        example_id=ex["id"], example_t=np.arange(len(ex["y"]), dtype=np.float32), example_y=ex["y"], example_sbar=ex["sc"][:, 1],
        example_a=ex["a"].astype(np.float32), example_w=ex["w"].astype(np.float32),
    )
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    summ = {name: {k: (rep[name][k]["ap"] if isinstance(rep[name].get(k), dict) else rep[name].get(k)) for k in ("base_rate", "s", "sbar", "a", "w")}
            for name in rep if rep[name]}
    print("[check1] AP by score (selected scorer):", json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
