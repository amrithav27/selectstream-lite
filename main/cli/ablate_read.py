"""Offline read-side ablations (docs/plan.md §13.5), CPU only, no trained weights.

On Check 2's closed-form memory (arm A cuts at the paper's B_s / quantile, tau from check2.json; nodes = mean
g_t of their frames), rank the nodes at each val question's t_q and report Recall@M: does any of the top M
nodes overlap the answer window? The scores are training-free, so this bounds what the read side can get
before learning and shows which pieces of Eq. 6 help:
  random      expected Recall@M of a uniform choice (exact, from the positive count)
  bias        the paper's fixed Eq. 6 terms alone: 0.2 * surprise - 0.05 * span - 0.05 * merges (the closed-form
              memory counts writes per node, not merges; write count - 1 stands in for the merge count)
  prior       zero-shot query-node cosine in the decoder's input-embedding space (VisMemModel.question_node_prior):
              mean content-word token embedding of the question - vocab mean vs node mean g - train mean
  prior+bias  z-scored prior + the paper's bias terms (the trained model's start, lam = 1)
Variants of the prior's text (query conditioning): question | object phrase (slot_x) | question with stopwords
kept | question, uncentred. Grids: M, N, open segment in / out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer

from main.cli.common import load_yaml
from main.cli.replay_write import Memory, cut_points, load_streams
from main.data.ego4d_qa import Ego4DStreamDataset
from main.data.frame_cache import FrameCache
from main.model.model import VisMemModel


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def norm_stat(x):
    x = np.clip(x, 0, None)
    return x / (x.max() + 1e-6)


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else np.zeros_like(x)


def random_recall(pos, m):
    n, p = len(pos), int(pos.sum())
    if m >= n:
        return 1.0
    return 1.0 - float(np.prod([(n - p - j) / (n - j) for j in range(m)]))


def bootstrap(x, n=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return [float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--feature_cache", required=True)
    ap.add_argument("--check2", required=True, help="replay/check2.json (arm A tau_r / tau_s).")
    ap.add_argument("--segment_center", required=True, help="segment_center.npy (train mean g_t) from Stage R.")
    ap.add_argument("--model_dir", default=None, help="Qwen2.5-VL snapshot dir (tokenizer + embed_tokens); default: the HF cache.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--budgets", default="8,16,32")
    ap.add_argument("--ms", default="1,2,4,8")
    ap.add_argument("--tracks", default="nlq,qa")
    ap.add_argument("--max_streams", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    v, d = cfg["vismem"], cfg["data"]
    fps, rho = float(d["sample_fps"]), float(v["lvm_saw_ema_decay"])
    P = dict(l_min=int(v["lvm_saw_l_min"]), l_max=int(v["lvm_saw_l_max"]), b_s=float(v["lvm_saw_energy_budget"]), q=float(v["lvm_saw_quantile"]),
             w=int(v["lvm_saw_recent_window"]))
    W = [float(v["method_eta_surprise_boost"]), float(v["method_span_penalty_weight"]), float(v["method_merge_penalty_weight"])]
    cal = json.load(open(args.check2))["calibration"]["A"]
    budgets, ms = [int(x) for x in args.budgets.split(",")], [int(x) for x in args.ms.split(",")]
    snap = args.model_dir or glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                                                    "hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*"))[0]
    tok = AutoTokenizer.from_pretrained(snap)
    shard = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]["model.embed_tokens.weight"]
    with safe_open(os.path.join(snap, shard), "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float().numpy()
    E_mu = E.mean(0)
    mu = np.load(args.segment_center)
    stop = VisMemModel.PRIOR_STOPWORDS

    def qvec(text, keep_stop=False, centre=True):
        words = [w.strip("?,.!").lower() for w in (text or "").split()]
        words = [w for w in words if w and (keep_stop or w not in stop)]
        ids = tok(" " + " ".join(words), add_special_tokens=False)["input_ids"] if words else []
        if not ids:
            return None
        e = E[ids].mean(0)
        return unit(e - E_mu) if centre else unit(e)

    os.makedirs(args.out_dir, exist_ok=True)
    cache = FrameCache(args.feature_cache)
    t0 = time.time()
    results = {"args": vars(args), "tracks": {}}
    for track in [t for t in args.tracks.split(",") if t]:
        src = {"nlq": "nlq", "qa": "qaego4d"}[track]
        ds = Ego4DStreamDataset(args.clean_dir, "", "val", src, part="all", sample_fps=fps, seed=int(cfg["training"]["seed"]), require_video=False,
                                query_time_mode=str(d.get("query_time_mode", "uniform")))
        ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)][: args.max_streams]
        ds.items = [ds[i] for i in range(len(ds.items))]  # fixed ask times, as Check 2
        streams = load_streams(ds, cache, fps, rho, 0.5, None, None)
        hits = {}

        def add(key, val):
            hits.setdefault(key, []).append(val)

        for st in streams:
            g, t, w = st["g"], st["t"], st["A"]
            cuts = set(cut_points(w, st["cut"], P["l_min"], P["l_max"], P["b_s"], P["q"], P["w"]))
            mems = {N: Memory(N, cal["tau_r"], cal["tau_s"], g.shape[1]) for N in budgets}
            a = 0
            pending = sorted(st["item"].queries, key=lambda q: q.ask_time)
            for i in range(len(t)):
                if i in cuts:
                    for mem in mems.values():
                        mem.write(g[a : i + 1].mean(0), i + 1 - a, float(np.mean(w[a : i + 1])), t[a], t[i])
                    a = i + 1
                while pending and (pending[0].ask_time <= t[i] + 1e-6 or i == len(t) - 1):
                    q = pending.pop(0)
                    texts = {"question": qvec(q.query), "object": qvec(q.slot_x) if q.slot_x else None,
                             "question+stopwords": qvec(q.query, keep_stop=True), "question, uncentred": qvec(q.query, centre=False)}
                    for N, mem in mems.items():
                        for open_in in (True, False):
                            G, sp, sur, mc = list(mem.g), list(zip(mem.t0, mem.t1)), list(mem.w), list(mem.wc - 1)
                            if open_in and a <= i:
                                G.append(g[a : i + 1].mean(0)); sp.append((t[a], t[i])); sur.append(float(np.mean(w[a : i + 1]))); mc.append(0.0)
                            if not G:
                                continue
                            sp = np.array(sp)
                            pos = (sp[:, 1] + 0.5 >= q.window_start - 0.5) & (sp[:, 0] - 0.5 <= q.window_end + 0.5)
                            if not pos.any():
                                continue
                            bias = W[0] * norm_stat(np.array(sur)) - W[1] * norm_stat(sp[:, 1] - sp[:, 0] + 1) - W[2] * norm_stat(np.array(mc))
                            Gc, Gr = unit(np.array(G) - mu), unit(np.array(G))
                            tag = f"N={N}|open={'in' if open_in else 'out'}"
                            for m in ms:
                                k = min(m, len(G))
                                top = lambda s: pos[np.argsort(-s, kind="stable")[:k]].any()
                                add(f"{tag}|M={m}|random", random_recall(pos, k))
                                add(f"{tag}|M={m}|bias", top(bias))
                                qv = texts["question"]
                                if qv is not None:
                                    pr = Gc @ qv
                                    add(f"{tag}|M={m}|prior: question", top(pr))
                                    add(f"{tag}|M={m}|prior+bias", top(zscore(pr) + bias))
                                    add(f"{tag}|M={m}|random (same questions)", random_recall(pos, k))
                                for name in ("object", "question+stopwords"):
                                    if texts[name] is not None:
                                        add(f"{tag}|M={m}|prior: {name}", top(Gc @ texts[name]))
                                if texts["question, uncentred"] is not None:
                                    add(f"{tag}|M={m}|prior: question, uncentred", top(Gr @ texts["question, uncentred"]))
        res = {k: {"recall": bootstrap(v), "n": len(v)} for k, v in hits.items()}
        results["tracks"][track] = {"streams": len(streams), "cells": res}
        for k in sorted(res):
            if "|M=4|" in k and "N=16|open=in" in k:
                r = res[k]["recall"]
                print(f"[read] {track} {k:55s} Recall {r[0]:.3f} [{r[1]:.3f},{r[2]:.3f}] n={res[k]['n']}", flush=True)
        print(f"[read] {track} done ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out_dir, "ablate_read.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
