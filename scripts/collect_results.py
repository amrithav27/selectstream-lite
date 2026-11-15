"""Collect a run's eval results with a test on every number: per-run means with 95% CIs, and paired
differences against a reference with a 95% CI and a p-value.

Unit of resampling = stream (video): questions in one stream share a memory, so they are not independent.
- Means: ratio of summed values to summed counts; CI from a bootstrap over streams.
- Paired differences (b - a, matched by query_id): per-stream summed differences d_s over counts c_s.
  CI: bootstrap over streams. p: two-sided sign-flip permutation test over streams (H0: the per-stream
  difference is symmetric around 0), statistic sum(d_s) / sum(c_s).
- Holm correction within each family (one reference, one metric), reported as p_holm.
Questions where either side has no value for a metric (e.g. Recall@M with M = 0) are dropped for that metric.
Every comparison is reported, significant or not; "sig" only marks p_holm < 0.05.

    python scripts/collect_results.py outputs/full_v3 outputs/full_v3/collected_results.json
Also writes collected_results.md (tables) next to the JSON.
"""
import json
import os
import sys

import numpy as np

MET = ["recall_at_m", "t_overlap", "best_iou", "evidence_precision", "contains", "exact", "num_committed"]
N_BOOT = 2000
N_PERM = 10000


def load(run_dir, d):
    p = os.path.join(run_dir, d, "predictions.jsonl")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {r["query_id"]: r for r in map(json.loads, f)}


def by_stream(rows, keys, val):
    """Sum val(k) and count per stream over keys; returns (sums, counts)."""
    idx, s, c = {}, [], []
    for k in keys:
        i = idx.setdefault(rows[k]["stream_id"], len(idx))
        if i == len(s):
            s.append(0.0)
            c.append(0)
        s[i] += val(k)
        c[i] += 1
    return np.array(s), np.array(c, dtype=float)


def boot_ratio(s, c, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def perm_p(d, c, seed=0):
    """Two-sided sign-flip permutation p over streams; +1 in numerator and denominator (never exactly 0)."""
    obs = abs(d.sum() / c.sum())
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = np.abs((signs * d).sum(1) / c.sum())
    return float((1 + np.sum(null >= obs - 1e-12)) / (N_PERM + 1))


def overall(rows, keep=None):
    keys = [k for k in rows if keep is None or keep(rows[k])]
    out = {"n": len(keys), "streams": len({rows[k]["stream_id"] for k in keys})}
    for m in MET:
        km = [k for k in keys if rows[k].get(m) is not None]
        if not km:
            continue
        s, c = by_stream(rows, km, lambda k: float(rows[k][m]))
        lo, hi = boot_ratio(s, c)
        out[m] = {"mean": float(s.sum() / c.sum()), "lo": lo, "hi": hi}
    return out


def paired(a, b, m, keep=None):
    q = [k for k in sorted(set(a) & set(b))
         if a[k].get(m) is not None and b[k].get(m) is not None and (keep is None or keep(a[k]))]
    if not q:
        return None
    d, c = by_stream(a, q, lambda k: float(b[k][m]) - float(a[k][m]))
    lo, hi = boot_ratio(d, c)
    return {"diff": float(d.sum() / c.sum()), "lo": lo, "hi": hi, "p": perm_p(d, c), "n": len(q), "streams": len(d)}


def same_prediction(a, b):
    q = set(a) & set(b)
    if not any(a[k].get("prediction") is not None for k in q):
        return None  # NLQ: no generated answer
    return float(np.mean([a[k]["prediction"] == b[k]["prediction"] for k in q]))


def holm(entries):
    """entries: list of dicts with 'p'; adds 'p_holm' and 'sig' in place."""
    ps = [e for e in entries if e is not None]
    order = sorted(range(len(ps)), key=lambda i: ps[i]["p"])
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]["p"]))
        ps[i]["p_holm"] = run
        ps[i]["sig"] = bool(run < 0.05)


def q2_stats(path):
    """Q2 per-item rows -> per-k means with CIs and paired (decomposed / unmatched - single) tests."""
    rows = [json.loads(x) for x in open(path)]
    out = {}
    for k in sorted({r["k"] for r in rows}):
        rs = [r for r in rows if r["k"] == k]
        streams = sorted({r["stream_id"] for r in rs})
        si = {s: i for i, s in enumerate(streams)}

        def agg(fn):
            s, c = np.zeros(len(streams)), np.zeros(len(streams))
            for r in rs:
                s[si[r["stream_id"]]] += fn(r)
                c[si[r["stream_id"]]] += 1
            return s, c
        ent = {"items": len(rs), "streams": len(streams)}
        arms = ["single", "decomposed", "unmatched", "survival"]
        for arm in arms:
            for name, fn in ((f"{arm}_subhit", lambda r, a=arm: float(np.mean(r[f"{a}_hits"]))),
                             (f"{arm}_allhit", lambda r, a=arm: float(all(r[f"{a}_hits"])))):
                s, c = agg(fn)
                lo, hi = boot_ratio(s, c)
                ent[name] = {"mean": float(s.sum() / c.sum()), "lo": lo, "hi": hi}
        fam = []
        for arm in ("decomposed", "unmatched"):
            for name, fn in ((f"{arm}-single_subhit", lambda r, a=arm: float(np.mean(r[f"{a}_hits"]) - np.mean(r["single_hits"]))),
                             (f"{arm}-single_allhit", lambda r, a=arm: float(all(r[f"{a}_hits"])) - float(all(r["single_hits"])))):
                d, c = agg(fn)
                lo, hi = boot_ratio(d, c)
                ent[name] = {"diff": float(d.sum() / c.sum()), "lo": lo, "hi": hi, "p": perm_p(d, c), "n": len(rs)}
                fam.append(ent[name])
        if "single_contains" in rs[0]:
            for arm in ("single", "decomposed"):
                s, c = agg(lambda r, a=arm: float(np.mean(r[f"{a}_contains"])))
                lo, hi = boot_ratio(s, c)
                ent[f"{arm}_contains"] = {"mean": float(s.sum() / c.sum()), "lo": lo, "hi": hi}
            d, c = agg(lambda r: float(np.mean(r["decomposed_contains"]) - np.mean(r["single_contains"])))
            lo, hi = boot_ratio(d, c)
            ent["decomposed-single_contains"] = {"diff": float(d.sum() / c.sum()), "lo": lo, "hi": hi, "p": perm_p(d, c), "n": len(rs)}
            fam.append(ent["decomposed-single_contains"])
        holm(fam)
        out[f"k{k}"] = ent
    return out


def groups_for(names):
    has = set(names)
    g = [
        ("eval_qa_A", ["eval_qa_B", "eval_qa_C", "eval_qa_D", "eval_qa_B_Rwriter", "eval_qa_untrained"]),
        ("eval_qa_B_Rwriter", ["eval_qa_B"]),
        ("eval_nlq_A", ["eval_nlq_B", "eval_nlq_C", "eval_nlq_D"]),
        ("read_full", sorted(d for d in names if d.startswith("read_") and d != "read_full")),
        ("eval_qa_A", ["read_full"]),  # read_full must reproduce the main eval on its streams
    ]
    # Memory budget N (AB-N1): every N run against the smallest N present.
    # eval_qa_A_N*: the original 9 long streams (N = 16 reference); budget_qa_A_N*: all val QA streams, and
    # eval_qa_A is N = 16 on those.
    ns = sorted((int(d[len("eval_qa_A_N"):]), d) for d in names if d.startswith("eval_qa_A_N") and d[len("eval_qa_A_N"):].isdigit())
    if len(ns) > 1:
        g.append((ns[0][1], [d for _, d in ns[1:]]))
    g.append(("eval_qa_A", sorted(d for d in names if d.startswith("budget_qa_A_N"))))
    # Seeds (AB-R1 / P4): B - A within each seed, and seed s - seed 42 per arm.
    for d in names:
        if d.startswith("eval_") and "_seed" in d:
            base, seed = d.rsplit("_seed", 1)
            track_arm = base[len("eval_"):]
            track, arm = track_arm.split("_", 1)
            if arm == "B" and f"eval_{track}_A_seed{seed}" in has:
                g.append((f"eval_{track}_A_seed{seed}", [d]))
            if base in has:
                g.append((base, [d]))
    return [(ref, [r for r in runs if r in has]) for ref, runs in g if ref in has]


def neg_delay_ok(r):
    return r.get("delay_bucket") != "<0s"


def main(run_dir, out_path):
    names = sorted(d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d)))
    out = {"method": __doc__.strip().split("\n\n")[0], "overall": {}, "overall_excl_neg_delay": {}, "diff": {},
           "diff_excl_neg_delay": {}, "q2": {}}
    cache = {}

    def get(d):
        if d not in cache:
            cache[d] = load(run_dir, d)
        return cache[d]
    for ref, runs in groups_for(names):
        a = get(ref)
        if a is None:
            continue
        out["overall"].setdefault(ref, overall(a))
        fams = {}
        for d in runs:
            b = get(d)
            if b is None:
                continue
            out["overall"].setdefault(d, overall(b))
            mets = [m for m in MET if m in next(iter(b.values())) and m in next(iter(a.values()))]
            diff = {m: paired(a, b, m) for m in mets}
            for m, e in diff.items():
                fams.setdefault(m, []).append(e)
            out["diff"][f"{d} - {ref}"] = diff | {"same_prediction": same_prediction(a, b)}
        for es in fams.values():
            holm(es)
    # Sensitivity check: the main arm comparisons without the questions asked before their window ends.
    for ref, runs in (("eval_qa_A", ["eval_qa_B", "eval_qa_C", "eval_qa_D"]), ("eval_nlq_A", ["eval_nlq_B", "eval_nlq_C", "eval_nlq_D"])):
        a = get(ref) if ref in names else None
        if a is None:
            continue
        out["overall_excl_neg_delay"][ref] = overall(a, neg_delay_ok)
        fams = {}
        for d in runs:
            b = get(d) if d in names else None
            if b is None:
                continue
            out["overall_excl_neg_delay"][d] = overall(b, neg_delay_ok)
            diff = {m: paired(a, b, m, neg_delay_ok) for m in MET if m in next(iter(b.values()))}
            for m, e in diff.items():
                fams.setdefault(m, []).append(e)
            out["diff_excl_neg_delay"][f"{d} - {ref}"] = diff
        for es in fams.values():
            holm(es)
    for d in names:
        if d.startswith("q2_"):
            f = [x for x in os.listdir(os.path.join(run_dir, d)) if x.endswith(".jsonl")]
            if f:
                out["q2"][d] = q2_stats(os.path.join(run_dir, d, f[0]))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    write_md(out, os.path.splitext(out_path)[0] + ".md")
    print(f"[collect] {len(out['overall'])} evals, {len(out['diff'])} paired diffs, {len(out['q2'])} Q2 runs -> {out_path}")


def fmt_ci(e, key="mean"):
    return "—" if e is None else f"{e[key]:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]" if key == "diff" else f"{e[key]:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


def fmt_p(e):
    if e is None:
        return "—"
    star = " *" if e.get("sig") else ""
    return f"{e['p']:.4f} / {e.get('p_holm', e['p']):.4f}{star}"


def write_md(out, path):
    L = ["# Collected results (generated by scripts/collect_results.py; do not edit by hand)", "",
         out["method"], "", "`*` = Holm-adjusted p < 0.05 within (reference, metric). Every comparison is listed.", "",
         "## Per-run means, 95% CI over streams", "",
         "| run | n | streams | Recall@M | best IoU | precision | contains | exact |", "|---|---|---|---|---|---|---|---|"]
    for d, o in out["overall"].items():
        L.append(f"| {d} | {o['n']} | {o['streams']} | " + " | ".join(fmt_ci(o.get(m)) for m in ("recall_at_m", "best_iou", "evidence_precision", "contains", "exact")) + " |")
    for title, key in (("Paired differences (b − a), 95% CI, p / p_holm", "diff"),
                       ("Same, excluding `<0s` delay questions", "diff_excl_neg_delay")):
        L += ["", f"## {title}", "", "| comparison | n | ΔRecall@M | p | Δbest IoU | p | Δcontains | p | same pred. |", "|---|---|---|---|---|---|---|---|---|"]
        for name, dd in out[key].items():
            n = next((e["n"] for e in dd.values() if isinstance(e, dict)), "—")
            cells = []
            for m in ("recall_at_m", "best_iou", "contains"):
                e = dd.get(m)
                cells += [fmt_ci(e, "diff"), fmt_p(e)]
            sp = dd.get("same_prediction")
            L.append(f"| {name} | {n} | " + " | ".join(cells) + f" | {'—' if sp is None else f'{sp:.1%}'} |")
    if out["q2"]:
        L += ["", "## Q2 (composite questions), 95% CI over streams", "",
              "| run | k | items | single all-hit | decomposed all-hit | Δ all-hit (p / p_holm) | Δ sub-hit (p / p_holm) | Δ contains (p / p_holm) |",
              "|---|---|---|---|---|---|---|---|"]
        for d, byk in out["q2"].items():
            for k, e in byk.items():
                dc = e.get("decomposed-single_contains")
                L.append(f"| {d} | {k[1:]} | {e['items']} | {fmt_ci(e['single_allhit'])} | {fmt_ci(e['decomposed_allhit'])} | "
                         f"{fmt_ci(e['decomposed-single_allhit'], 'diff')} ({fmt_p(e['decomposed-single_allhit'])}) | "
                         f"{fmt_ci(e['decomposed-single_subhit'], 'diff')} ({fmt_p(e['decomposed-single_subhit'])}) | "
                         f"{'—' if dc is None else fmt_ci(dc, 'diff') + ' (' + fmt_p(dc) + ')'} |")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
