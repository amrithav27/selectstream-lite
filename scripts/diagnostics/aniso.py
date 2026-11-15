"""How parallel are the cached g_t? Cosine between frames of different videos and within a video,
raw vs centred on the train mean (a fixed vector, computable once from the cache)."""
import glob, json, os, sys
import numpy as np
root = sys.argv[1]
rng = np.random.default_rng(0)
clips = json.load(open("dataset/clean/clips.json"))
tr = [c["clip_uid"] for c in clips if c["split"] == "train" and os.path.exists(f"{root}/g/{c['clip_uid']}.npy")]
va = [c["clip_uid"] for c in clips if c["split"] == "val" and os.path.exists(f"{root}/g/{c['clip_uid']}.npy")]
def sample(uids, per=40):
    out = []
    for u in uids:
        g = np.load(f"{root}/g/{u}.npy", mmap_mode="r")
        idx = rng.choice(len(g), min(per, len(g)), replace=False)
        out.append(np.asarray(g[np.sort(idx)], dtype=np.float64))
    return out
trs = sample(rng.choice(tr, 400, replace=False))
mu = np.concatenate(trs).mean(0)
vas = sample(rng.choice(va, 150, replace=False))
def unit(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
def stats(groups, c):
    X = [unit(g - c) for g in groups]
    within = np.mean([ (x @ x.T)[np.triu_indices(len(x), 1)].mean() for x in X])
    A = np.concatenate([x[:5] for x in X]); vid = np.repeat(np.arange(len(X)), [min(5, len(x)) for x in X])
    S = A @ A.T; diff = vid[:, None] != vid[None]
    near = []  # adjacent frames, 1 s apart, within a clip (sampled frames are sorted, not adjacent: use raw files)
    return within, S[diff].mean(), np.quantile(S[diff], [0.05, 0.95])
for name, c in (("raw", np.zeros_like(mu)), ("centred on train mean", mu)):
    w, b, q = stats(vas, c)
    print(f"{name:22s} cos within a video {w:.3f} | across videos {b:.3f} (q05 {q[0]:.3f}, q95 {q[1]:.3f})")
print("norm of train mean / mean norm of g:", round(float(np.linalg.norm(mu) / np.mean([np.linalg.norm(g, axis=1).mean() for g in trs])), 3))
