"""Node-level: Recall@M (M = 4) at t_q when nodes are ranked by the zero-shot score
cos(question token embedding - vocab mean, node mean g - train mean), vs random choice and vs the paper's
bias terms alone (eta * surprise - span - merge, normalised as dmg_gar). Closed-form memory of Check 2
(arm A cuts, paper B_s / quantile, tau from check2.json), committed nodes + the open segment, N = 16.
"""
import glob
import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from streamlite.cli.common import load_yaml
from streamlite.cli.replay_write import Memory, cut_points, load_streams
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache

CACHE, CENTER, CHECK2 = sys.argv[1:4]
M, N = 4, 16
snap = glob.glob("/ssd_scratch/amritha.vinod/selectstream/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap)
with safe_open(f"{snap}/model-00001-of-00005.safetensors", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float().numpy()
E_mu = E.mean(0)
STOP = set("the a an i my me did do does what where when how who which was were is are of to in on at for with from it this that "
           "before after there many much put get take see saw".split())
mu = np.load(CENTER)
cal = json.load(open(CHECK2))["calibration"]["A"]
v = load_yaml("configs/ego4d_qwen25vl7b.yaml")["vismem"]
P = dict(l_min=int(v["lvm_saw_l_min"]), l_max=int(v["lvm_saw_l_max"]), b_s=float(v["lvm_saw_energy_budget"]),
         q=float(v["lvm_saw_quantile"]), w=int(v["lvm_saw_recent_window"]))
rng = np.random.default_rng(0)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def qemb(text):
    words = [w.strip("?,.!").lower() for w in text.split()]
    words = [w for w in words if w and w not in STOP]
    ids = tok(" " + " ".join(words), add_special_tokens=False)["input_ids"] if words else []
    return unit(E[ids].mean(0) - E_mu) if ids else None


cache = FrameCache(CACHE)
ds = Ego4DStreamDataset("dataset/clean", "", "val", "nlq", part="all", sample_fps=1.0, seed=42, require_video=False)
ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
ds.items = [ds[i] for i in range(len(ds.items))]  # fixed ask times, as Check 2
streams = load_streams(ds, cache, 1.0, float(v["lvm_saw_ema_decay"]), 0.5, None, None)

hits = {"zero-shot": [], "random": [], "bias only (paper Eq. 6 terms)": [], "zero-shot + bias": [], "14.3 * zero-shot + bias": [], "zero-shot, no open segment": []}
cands = []
spreads = []
for st in streams:
    g, t, w = st["g"], st["t"], st["A"]
    cuts = set(cut_points(w, st["cut"], P["l_min"], P["l_max"], P["b_s"], P["q"], P["w"]))
    mem = Memory(N, cal["tau_r"], cal["tau_s"], g.shape[1])
    a = 0
    pending = sorted(st["item"].queries, key=lambda q: q.ask_time)
    for i in range(len(t)):
        if i in cuts:
            mem.write(g[a : i + 1].mean(0), i + 1 - a, float(np.mean(w[a : i + 1])), t[a], t[i])
            a = i + 1
        while pending and (pending[0].ask_time <= t[i] + 1e-6 or i == len(t) - 1):
            q = pending.pop(0)
            nodes_g = list(mem.g)
            spans = list(zip(mem.t0, mem.t1))
            sur = list(mem.w)
            cnt = list(mem.wc)
            if a <= i:  # open segment as a provisional node
                nodes_g.append(g[a : i + 1].mean(0)); spans.append((t[a], t[i])); sur.append(float(np.mean(w[a : i + 1]))); cnt.append(1.0)
            if not nodes_g:
                continue
            sp = np.array(spans)
            pos = (sp[:, 1] + 0.5 >= q.window_start - 0.5) & (sp[:, 0] - 0.5 <= q.window_end + 0.5)
            if not pos.any():
                continue
            e = qemb(q.query)
            if e is None:
                continue
            zs = unit(np.array(nodes_g) - mu) @ e
            spreads.append(np.quantile(zs, .9) - np.quantile(zs, .1))
            nz = lambda x: np.clip(x, 0, None) / (np.clip(x, 0, None).max() + 1e-6)
            span_len = sp[:, 1] - sp[:, 0] + 1
            bias = 0.2 * nz(np.array(sur)) - 0.05 * nz(span_len) - 0.05 * nz(np.array(cnt) - 1)
            k = min(M, len(zs))
            top = lambda s: np.argsort(-s)[:k]
            hits["zero-shot"].append(pos[top(zs)].any())
            hits["bias only (paper Eq. 6 terms)"].append(pos[top(bias + 1e-9 * rng.random(len(bias)))].any())
            hits["zero-shot + bias"].append(pos[top(zs + bias)].any())
            hits["14.3 * zero-shot + bias"].append(pos[top(14.2857 * zs + bias)].any())
            nc = len(mem.g)
            if nc >= 1 and pos[:nc].any():
                kk = min(M, nc)
                hits["zero-shot, no open segment"].append(pos[:nc][np.argsort(-zs[:nc])[:kk]].any())
            hits["random"].append(1 - np.prod([(len(pos) - pos.sum() - j) / (len(pos) - j) for j in range(k)]))
            cands.append(len(pos))
print("zero-shot score spread (q90 - q10 over a question's nodes), median:", round(float(np.median(spreads)), 4))
print(f"{len(cands)} questions with a positive node; candidates per question: median {np.median(cands):.0f}")
for name, h in hits.items():
    h = np.array(h, float)
    b = [rng.choice(h, len(h)).mean() for _ in range(1000)]
    print(f"Recall@{M} {name:32s} {h.mean():.3f} [{np.quantile(b, .025):.3f},{np.quantile(b, .975):.3f}]")
