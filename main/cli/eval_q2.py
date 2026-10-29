"""Q2: decomposed vs single-step retrieval at equal evidence budget M (docs/plan.md §7).

Items, per val video stream, built from that video's own queries (one track at a time):
  T1 conjunction, k = 2 and 3: sub-questions whose windows are >= --min_gap s apart (video time), so the hops
     are separate moments in memory; <= --per_video items per video per k, drawn with a fixed seed.
  T2 order (k = 2 items): "which happened first?" is decidable from the two windows.
Each item is asked at a frame drawn uniformly at or after its last window end (same rule as the tracks,
seeded per item). All arms read the same memory snapshot (reads do not mark read counts):
  single      one read on the composite question, top-M
  decomposed  one read per sub-question, M split as evenly as possible (total exactly M)   <- the result
  unmatched   one read per sub-question, top-M each (k*M in total)                          <- diagnostic only
Metrics: SubHit (fraction of sub-question windows hit by that sub-question's evidence; for single, by any
of its evidence), AllHit (every window hit), T2 order accuracy from the decomposed reads' evidence times,
the evidence-check curve (a sub-question abstains when its top refined GAR score is below a threshold:
coverage vs. unsupported rate = answered sub-questions whose evidence misses their window), and on the QA
track answer contains-match per sub-question (single: the gold sub-answer inside the one composite answer).
Limitation: the split is the oracle split the item was built from; it bounds what a decomposer can do.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np
import torch

from main.cli.common import load_yaml
from main.cli.ego4d_common import apply_arm, format_prompt, load_model, log_device, prepare_scorer
from main.data.ego4d_qa import Ego4DStreamDataset, _stable_unit, sample_ask_time
from main.data.frame_cache import FrameCache, as_image, stream_inputs
from main.trainer.method_sft import _temporal_overlap

ANSWER_EACH = "Answer each question separately, numbered.\n"


def normalize(s):
    import re
    import string

    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def build_items(queries, k, min_gap, per_video, seed, stream_id):
    """All k-subsets whose windows are pairwise >= min_gap apart, then <= per_video of them (seeded)."""
    qs = sorted(queries, key=lambda q: q.window_start)
    sep = lambda a, b: b.window_start - a.window_end >= min_gap or a.window_start - b.window_end >= min_gap
    cands = []
    for combo in combinations(range(len(qs)), k):
        if all(sep(qs[i], qs[j]) for i, j in combinations(combo, 2)):
            cands.append([qs[i] for i in combo])
        if len(cands) >= 5000:
            break
    cands.sort(key=lambda g: _stable_unit(seed, stream_id, k, *[q.query_id for q in g]))
    return cands[:per_video]


def composite(group):
    return ANSWER_EACH + " ".join(f"({i + 1}) {q.query.strip()}" for i, q in enumerate(group))


def read(model, img, question, state, k_evidence, t_now):
    inputs = model.prepare_inputs([img], [format_prompt(question)])
    out = model.base_model(**inputs, output_hidden_states=True)
    _, aux = model.read_evidence(out.hidden_states[-1], inputs["input_ids"], inputs.get("attention_mask"), state,
                                 mark_read=False, evidence_k=k_evidence, t_now=t_now)
    if not aux:
        return torch.empty(0, 2), float("-inf")
    spans = aux.get("evidence_time_spans", torch.empty(0, 2)).float().cpu()
    scores = aux.get("node_scores")
    return spans, float(scores.max()) if scores is not None and scores.numel() else float("-inf")


def hit(spans, window):
    return float(spans.numel() > 0 and _temporal_overlap(spans, [window], 1.0) > 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--torch_dtype", default=None)
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--track", choices=["nlq", "qa"], default="nlq")
    ap.add_argument("--ks", default="2,3")
    ap.add_argument("--min_gap", type=float, default=30.0)
    ap.add_argument("--per_video", type=int, default=3)
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--feature_cache", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_clips", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--calibration", default="outputs/calibration.json")
    args = ap.parse_args()

    cfg_dict = apply_arm(load_yaml(args.config), args.arm, args.calibration)
    if args.torch_dtype:
        cfg_dict["model"]["torch_dtype"] = args.torch_dtype
    dcfg, tcfg = cfg_dict["data"], cfg_dict["training"]
    m_total = int(cfg_dict["vismem"]["method_evidence_tokens"])
    seed = int(tcfg.get("seed", 42))
    os.makedirs(args.out_dir, exist_ok=True)
    log_device()
    model = load_model(cfg_dict, ckpt=args.ckpt)
    prepare_scorer(model, args.arm, args.scorer)
    model.eval()
    sample_fps = float(dcfg.get("sample_fps", 1.0))
    cache = FrameCache(args.feature_cache, args.video_dir, sample_fps, int(dcfg.get("frame_max_side", 448))) if args.feature_cache else None
    ds = Ego4DStreamDataset(args.clean_dir, args.video_dir, "val", "qaego4d" if args.track == "qa" else "nlq",
                            query_time_mode=str(dcfg.get("query_time_mode", "uniform")), sample_fps=sample_fps,
                            resample_ask_times=False, seed=seed, require_video=cache is None)
    if cache is not None:
        ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
    if args.max_clips is not None:
        ds.items = ds.items[: args.max_clips]
    ks = [int(x) for x in args.ks.split(",")]

    rows = []
    with open(os.path.join(args.out_dir, f"q2_{args.track}.jsonl"), "w") as fout, torch.no_grad():
        for item in ds.items:
            sched = []
            for k in ks:
                for g in build_items(item.queries, k, args.min_gap, args.per_video, seed, item.stream_id):
                    key = "|".join(sorted(q.query_id for q in g))
                    sched.append((sample_ask_time(max(q.window_end for q in g), item.frame_times, ds.query_time_mode, _stable_unit(seed, 0, key)), g))
            sched.sort(key=lambda x: x[0])
            n_items = len(sched)
            state = model.init_memory_state(batch_size=1, dtype=model.trainable_dtype, device=model.trainable_device)
            for img, gt, t, cut, last in stream_inputs(item, sample_fps, int(dcfg.get("frame_max_side", 448)), cache):
                state, _ = model.update_memory_frame(img, t, state, flush=cut, g=gt)
                while sched and (sched[0][0] <= t + 1e-6 or last):
                    t_q, g = sched.pop(0)
                    k = len(g)
                    image = as_image(img)
                    windows = [(q.window_start, q.window_end) for q in g]
                    split_m = [m_total // k + (1 if j < m_total % k else 0) for j in range(k)]
                    s_spans, s_score = read(model, image, composite(g), state, m_total, t)
                    dec = [read(model, image, q.query, state, mj, t) for q, mj in zip(g, split_m)]
                    unm = [read(model, image, q.query, state, m_total, t) for q in g]
                    view = model.lvm_short_former.state_view(state[0], include_open=True, t_now=t)
                    committed = torch.stack([view["active_time_starts"], view["active_time_ends"]], -1).float().cpu()
                    row = {
                        "stream_id": item.stream_id, "query_ids": [q.query_id for q in g], "k": k, "ask_time": t_q,
                        "evidence_budget": {"single": m_total, "decomposed": split_m, "unmatched": [m_total] * k},
                        "single_hits": [hit(s_spans, w) for w in windows],
                        "decomposed_hits": [hit(sp, w) for (sp, _), w in zip(dec, windows)],
                        "unmatched_hits": [hit(sp, w) for (sp, _), w in zip(unm, windows)],
                        "survival_hits": [hit(committed, w) for w in windows],
                        "decomposed_top_score": [sc for _, sc in dec],
                        "single_top_score": s_score,
                    }
                    if k == 2:
                        true_first = 0 if g[0].window_start < g[1].window_start else 1
                        centers = [float(sp.mean()) if sp.numel() else float("nan") for sp, _ in dec]
                        row["order_true_first"] = true_first
                        row["order_pred_first"] = None if any(np.isnan(centers)) else int(np.argmin(centers))
                    if args.track == "qa":
                        pred_single, _ = model.answer_with_memory(image, format_prompt(composite(g)), state, max_new_tokens=args.max_new_tokens * k,
                                                                  mark_read=False, t_now=t, evidence_k=m_total)
                        preds = [model.answer_with_memory(image, format_prompt(q.query), state, max_new_tokens=args.max_new_tokens,
                                                          mark_read=False, t_now=t, evidence_k=mj)[0] for q, mj in zip(g, split_m)]
                        row["answers"] = [q.answer for q in g]
                        row["single_prediction"], row["decomposed_predictions"] = pred_single, preds
                        row["single_contains"] = [float(bool(normalize(q.answer)) and normalize(q.answer) in normalize(pred_single)) for q in g]
                        row["decomposed_contains"] = [float(bool(normalize(q.answer)) and (normalize(q.answer) in normalize(p) or (bool(normalize(p)) and normalize(p) in normalize(q.answer))))
                                                      for q, p in zip(g, preds)]
                    rows.append(row)
                    fout.write(json.dumps(row) + "\n")
            print(f"[q2] {item.stream_id}: {n_items} items", flush=True)

    summary = {"arm": args.arm, "track": args.track, "M": m_total, "min_gap_s": args.min_gap, "per_video": args.per_video, "ckpt": args.ckpt, "by_k": {}}
    for k in ks:
        rs = [r for r in rows if r["k"] == k]
        if not rs:
            continue
        ent = {"items": len(rs), "streams": len({r["stream_id"] for r in rs})}
        for arm in ("single", "decomposed", "unmatched", "survival"):
            ent[f"{arm}_subhit"] = float(np.mean([h for r in rs for h in r[f"{arm}_hits"]]))
            ent[f"{arm}_allhit"] = float(np.mean([float(all(r[f"{arm}_hits"])) for r in rs]))
        if k == 2:
            ok = [r for r in rs if r.get("order_pred_first") is not None]
            ent["order_acc_decomposed"] = float(np.mean([r["order_pred_first"] == r["order_true_first"] for r in ok])) if ok else None
        sc = np.array([s for r in rs for s in r["decomposed_top_score"]])
        hits = np.array([h for r in rs for h in r["decomposed_hits"]])
        curve = []
        for qn in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            thr = float(np.quantile(sc, qn))
            ans = sc >= thr
            curve.append({"threshold_quantile": qn, "threshold": thr, "coverage": float(ans.mean()),
                          "unsupported_rate": float((ans & (hits == 0)).sum() / max(ans.sum(), 1))})
        ent["evidence_check_curve"] = curve
        if args.track == "qa":
            ent["single_contains"] = float(np.mean([c for r in rs for c in r["single_contains"]]))
            ent["decomposed_contains"] = float(np.mean([c for r in rs for c in r["decomposed_contains"]]))
            ent["single_all_correct"] = float(np.mean([float(all(r["single_contains"])) for r in rs]))
            ent["decomposed_all_correct"] = float(np.mean([float(all(r["decomposed_contains"])) for r in rs]))
        summary["by_k"][str(k)] = ent
    summary["config"] = cfg_dict
    with open(os.path.join(args.out_dir, f"q2_{args.track}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
