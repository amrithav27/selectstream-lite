"""Zero-shot: do Qwen's own text token embeddings point at the frames a question asks about?

g_t (cached) is the vision projector output, mean-pooled: it lives in the decoder's input-embedding space,
the same space as text token embeddings. For each val NLQ query, embed the question (mean of its content
token embeddings, and separately the slot_x object phrase when present), centre both sides, and rank every
frame of the stream by cosine. Within-stream AUC of answer frames vs the rest; control: the same query
scored against another query's window in the same stream (shifted labels).
"""
import glob
import json
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from streamlite.cli.ego4d_common import frame_labels
from streamlite.cli.train_scorer import auroc
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache

CACHE = sys.argv[1]
snap = glob.glob("/ssd_scratch/amritha.vinod/selectstream/hf_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(snap)
with safe_open(f"{snap}/model-00001-of-00005.safetensors", "pt") as f:
    E = f.get_tensor("model.embed_tokens.weight").float()
print("embed", tuple(E.shape))
STOP = set("the a an i my me did do does what where when how who which was were is are of to in on at for with from it this that "
           "before after after? there many much put get take see saw".split())

cache = FrameCache(CACHE)
ds = Ego4DStreamDataset("dataset/clean", "", "val", "nlq", part="all", sample_fps=1.0, seed=42, require_video=False)
ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
mu = np.load(sys.argv[2])  # train mean g (segment_center.npy)
rng = np.random.default_rng(0)


def emb(text):
    words = [w.strip("?,.!").lower() for w in text.split()]
    words = [w for w in words if w and w not in STOP]
    if not words:
        return None
    ids = tok(" " + " ".join(words), add_special_tokens=False)["input_ids"]
    return E[ids].mean(0).numpy()


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


E_mu = E.mean(0).numpy()  # centre text on the vocabulary mean, frames on the train-frame mean
res = {"question": [], "object": [], "question_raw": [], "control": []}
for item in ds.items:
    g, t, _ = cache.stream_g(item, 1.0)
    G = unit(g - mu)
    Graw = unit(g)
    qs = item.queries
    for qi, q in enumerate(qs):
        y = frame_labels(t, [(q.window_start, q.window_end)], 0.5)
        if not (0 < y.sum() < len(y)):
            continue
        e = emb(q.query)
        if e is None:
            continue
        res["question"].append(auroc(G @ unit(e - E_mu), y))
        res["question_raw"].append(auroc(Graw @ unit(e), y))
        if q.slot_x:
            eo = emb(q.slot_x)
            if eo is not None:
                res["object"].append(auroc(G @ unit(eo - E_mu), y))
        other = [o for o in qs if o is not q and abs(o.window_start - q.window_start) > 30]
        if other:
            o = other[rng.integers(len(other))]
            yo = frame_labels(t, [(o.window_start, o.window_end)], 0.5)
            if 0 < yo.sum() < len(yo):
                res["control"].append(auroc(G @ unit(e - E_mu), yo))
for k, v in res.items():
    v = np.array(v)
    boot = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    print(f"{k:13s} n={len(v):5d} mean within-stream AUC {v.mean():.3f} [{np.quantile(boot, .025):.3f},{np.quantile(boot, .975):.3f}]  "
          f"share > 0.6: {(v > 0.6).mean():.2f}")
