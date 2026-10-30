"""Evaluate one arm on one track (qa = QaEgo4D val, nlq = NLQ val), strictly online (docs/plan.md §1).

Writes predictions.jsonl (one row per query, raw prediction kept so any answer metric can be
recomputed later) and summary.json (overall + per delay bucket / linked / participant-disjoint).
"""
from __future__ import annotations
import argparse
import json
import os
import re
import string
import time
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from streamlite.cli.common import load_yaml
from streamlite.cli.ego4d_common import apply_arm, delay_bucket, format_prompt, frame_labels, load_model, log_device, prepare_scorer, trace_extras, train_participants
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache, as_image, stream_inputs
from streamlite.trainer.method_sft import evidence_location, retrieval_metrics
from streamlite.utils.misc import set_seed
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
def answer_scores(pred: str, gold: str) -> dict:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return {"exact": float(p == g), "contains": float(bool(g) and (g in p or (bool(p) and p in g)))}
def summarize(rows, keys):
    out = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--torch_dtype", default=None, help="Override model.torch_dtype (bfloat16 | float16 | float32).")
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--ckpt", required=True, help="'none' = untrained trainables (seed-42 init): the no-learning baseline.")
    ap.add_argument("--track", choices=["qa", "nlq"], required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val", help="train = the dev participants (early stopping).")
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start", type=int, default=0, help="Skip the first K streams (with --max_clips, evaluates streams K..K+N-1).")
    ap.add_argument("--max_clips", type=int, default=None, help="First N streams (after --start).")
    ap.add_argument("--min_frames", type=int, default=0, help="Keep only streams with at least this many frames (memory-budget ablation).")
    ap.add_argument("--num_slots", type=int, default=None, help="Override N (lvm_num_slots); slot_init is tiled from the checkpoint's.")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--trace_dir", default=None)
    ap.add_argument("--trace_clips", type=int, default=3)
    ap.add_argument("--calibration", default="outputs/calibration.json", help="calibrate_write output; overrides the config arms block for B/C/D.")
    ap.add_argument("--feature_cache", default=None, help="Stage-0 cache dir: write path reads g_t, queries read the cached frame.")
    ap.add_argument("--scorer", default=None, help="Stage-1 scorer (train_scorer.py); required for arm B.")
    ap.add_argument("--read_ablation", default="none",
                    choices=["none", "no_prior", "no_cos", "no_bias", "cos_only", "prior_only", "no_hops", "no_gar", "coarse_only", "random", "oracle"],
                    help="Read-side ablation at eval time (dmg_gar.py; docs/plan.md §13.5).")
    ap.add_argument("--evidence_k", type=int, default=None, help="Evidence tokens M at read (0 = answer without memory).")
    ap.add_argument("--no_open", action="store_true", help="The read sees committed nodes only, not the open segment.")
    ap.add_argument("--blank_frame", action="store_true", help="Query-conditioning ablation: ask over a black frame of the same size.")
    ap.add_argument("--frames_baseline", choices=["none", "oracle", "uniform"], default="none",
                    help="P3 baseline (plain Qwen without memory): sample oracle frames (from answer window) or uniform frames (0 to t_q).")
    ap.add_argument("--num_frames", type=int, default=8, help="Number of frames to sample for --frames_baseline (default: 8).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume evaluation from existing predictions.jsonl, skipping completed streams.")
    ap.add_argument("--segment_center", default=None, help="With --ckpt none: a segment_center.npy (train_ego4d.py) so the "
                    "untrained baseline centres like the trained run; a checkpoint carries its own.")
    args = ap.parse_args()
    cfg_dict = apply_arm(load_yaml(args.config), args.arm, args.calibration)
    if args.torch_dtype:
        cfg_dict["model"]["torch_dtype"] = args.torch_dtype
    if args.num_slots is not None:
        cfg_dict["vismem"]["lvm_num_slots"] = args.num_slots
    dcfg, tcfg = cfg_dict["data"], cfg_dict["training"]
    set_seed(int(tcfg.get("seed", 42)))  # fixes the untrained init for --ckpt none
    os.makedirs(args.out_dir, exist_ok=True)
    log_device()
    model = load_model(cfg_dict, ckpt=None if args.ckpt == "none" else args.ckpt)
    prepare_scorer(model, args.arm, args.scorer)
    if args.segment_center:
        if args.ckpt != "none":
            raise ValueError("--segment_center is for --ckpt none; a checkpoint carries its own segment_center")
        model.lvm_short_former.set_segment_center(torch.from_numpy(np.load(args.segment_center)))
    model.graph_reasoner.read_ablation = args.read_ablation
    model.eval()
    sample_fps = float(dcfg.get("sample_fps", 1.0))
    cache = FrameCache(args.feature_cache, args.video_dir, sample_fps, int(dcfg.get("frame_max_side", 448))) if args.feature_cache else None
    source = "qaego4d" if args.track == "qa" else "nlq"
    ds = Ego4DStreamDataset(
        clean_dir=args.clean_dir,
        video_dir=args.video_dir,
        split=args.split,
        source=source,
        part="dev" if args.split == "train" else "all",
        dev_frac=float(dcfg.get("dev_frac", 0.1)),
        query_time_mode=str(dcfg.get("query_time_mode", "uniform")),
        sample_fps=sample_fps,
        resample_ask_times=False,
        seed=int(tcfg.get("seed", 42)),
        require_video=cache is None,
    )
    if cache is not None:
        n0 = len(ds.items)
        ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
        print(f"[eval] feature cache {args.feature_cache}: {len(ds.items)}/{n0} streams fully cached")
    if args.min_frames:
        ds.items = [i for i in ds.items if i.num_frames >= args.min_frames]
    ds.items = ds.items[args.start:]
    if args.max_clips is not None:
        ds.items = ds.items[: args.max_clips]
    seen_participants = train_participants(args.clean_dir)
    # A question is "linked" when it exists in both tracks (QaEgo4D row joined to an NLQ row).
    with open(os.path.join(args.clean_dir, f"qaego4d_{args.split}.json")) as f:
        linked_ids = {r["nlq_query_id"] for r in json.load(f) if r["nlq_query_id"] is not None}
    tol = float(tcfg.get("time_tolerance", 0.5))
    oracle = model.cfg.lvm_write_signal == "oracle"
    print(f"[eval] arm={args.arm} track={args.track} split={args.split} streams={len(ds)} missing_video={len(ds.missing_video)}")
    pred_path = os.path.join(args.out_dir, "predictions.jsonl")
    rows = []
    seen_stream_ids = set()
    mode = "w"
    if args.resume and os.path.exists(pred_path):
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows.append(r)
                    if "stream_id" in r:
                        seen_stream_ids.add(r["stream_id"])
        mode = "a"
        if seen_stream_ids:
            print(f"[eval] --resume: found {len(rows)} existing predictions across {len(seen_stream_ids)} streams in {pred_path}")
    with open(pred_path, mode) as fout, torch.no_grad():
        for ci, item in enumerate(ds):
            if args.resume and item.stream_id in seen_stream_ids:
                continue
            t0 = time.time()
            if args.frames_baseline != "none":
                seq = cache.stream_index(item, sample_fps) if cache else []
                for q in item.queries:
                    window = (q.window_start, q.window_end)
                    t_q = q.ask_time
                    if args.frames_baseline == "oracle":
                        cands = [x for x in seq if window[0] <= x[2] <= window[1]]
                        if not cands and seq:
                            mid = 0.5 * (window[0] + window[1])
                            cands = [min(seq, key=lambda x: abs(x[2] - mid))]
                    else:  # uniform
                        cands = [x for x in seq if x[2] <= t_q + 1e-6]
                        if not cands and seq:
                            cands = [seq[0]]
                    if len(cands) <= args.num_frames:
                        sel = cands
                    else:
                        idxs = np.round(np.linspace(0, len(cands) - 1, args.num_frames)).astype(int)
                        sel = [cands[i] for i in idxs]
                    images = [cache.image(u, k) for u, k, _ in sel] if cache else []
                    if not images:
                        images = [Image.new("RGB", (448, 448))]
                    if args.track == "qa":
                        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": format_prompt(q.query)}]
                        chat_text = model.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                        inputs = model.processor(text=[chat_text], images=images, return_tensors="pt", padding=True)
                        inputs = {k: v.to(model.trainable_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                        gen = model.base_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
                        pred = model.processor.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                    else:
                        pred = None
                    spans = [[t, t + 1.0 / sample_fps] for _, _, t in sel]
                    spans_tensor = torch.tensor(spans, dtype=torch.float32) if spans else torch.empty(0, 2)
                    aux = {"evidence_time_spans": spans_tensor}
                    committed = spans_tensor
                    row = {
                        "query_id": q.query_id,
                        "stream_id": item.stream_id,
                        "clip_uid": q.clip_uid,
                        "track": args.track,
                        "arm": f"frames_{args.frames_baseline}",
                        "template": q.template,
                        "linked": q.nlq_query_id in linked_ids,
                        "nlq_query_id": q.nlq_query_id,
                        "participant_disjoint": item.participant_id is not None and item.participant_id not in seen_participants,
                        "window": list(window),
                        "ask_time": q.ask_time,
                        "delay_bucket": delay_bucket(q),
                        "evidence_location": "baseline_frames",
                        "question": q.query,
                        "answer": q.answer,
                        "prediction": pred,
                        "evidence_spans": spans,
                    }
                    row.update(retrieval_metrics(aux, committed, window, tol))
                    if pred is not None and q.answer:
                        row.update(answer_scores(pred, q.answer))
                    rows.append(row)
                    fout.write(json.dumps(row) + "\n")
                print(f"[eval] stream {ci + 1}/{len(ds)} {item.stream_id} clips={len(item.pieces)} queries={len(item.queries)} {time.time() - t0:.0f}s (baseline={args.frames_baseline})", flush=True)
                continue
            state = model.init_memory_state(batch_size=1, dtype=model.trainable_dtype, device=model.trainable_device)
            windows = [(q.window_start, q.window_end) for q in item.queries]
            pending = list(item.queries)
            trace, n_frames = [], 0
            for img, g, t, cut, last in stream_inputs(item, sample_fps, int(dcfg.get("frame_max_side", 448)), cache):
                n_frames += 1
                y_t = float(frame_labels([t], windows, tol)[0])
                state, info = model.update_memory_frame(img, t, state, flush=cut, frame_signal=y_t if oracle else None, g=g)
                if args.trace_dir and ci < args.trace_clips:
                    trace.append({k: (v.tolist() if torch.is_tensor(v) else v) for k, v in info.items() if k != "importance_logit"}
                                 | {"importance": float(torch.sigmoid(info["importance_logit"])) if info.get("importance_logit") is not None else None}
                                 | trace_extras(model, state, img, g, n_frames - 1))
                while pending and (pending[0].ask_time <= t + 1e-6 or last):
                    q = pending.pop(0)
                    view = model.lvm_short_former.state_view(state[0], include_open=True, t_now=t)
                    committed = torch.stack([view["active_time_starts"], view["active_time_ends"]], dim=-1)
                    window = (q.window_start, q.window_end)
                    img_q = as_image(img)
                    if args.blank_frame:
                        img_q = Image.new("RGB", img_q.size)
                    model.graph_reasoner.oracle_window = window
                    t_read = None if args.no_open else t
                    if args.track == "qa":
                        pred, aux = model.answer_with_memory(img_q, format_prompt(q.query), state, max_new_tokens=args.max_new_tokens,
                                                             t_now=t_read, evidence_k=args.evidence_k)
                    else:
                        inputs = model.prepare_inputs([img_q], [format_prompt(q.query)])
                        out = model.base_model(**inputs, output_hidden_states=True)
                        _, aux = model.read_evidence(out.hidden_states[-1], inputs["input_ids"], inputs.get("attention_mask"), state,
                                                     t_now=t_read, evidence_k=args.evidence_k)
                        pred = None
                    row = {
                        "query_id": q.query_id,
                        "stream_id": item.stream_id,
                        "clip_uid": q.clip_uid,
                        "track": args.track,
                        "arm": args.arm,
                        "template": q.template,
                        "linked": q.nlq_query_id in linked_ids,
                        "nlq_query_id": q.nlq_query_id,
                        "participant_disjoint": item.participant_id is not None and item.participant_id not in seen_participants,
                        "window": list(window),
                        "ask_time": q.ask_time,
                        "delay_bucket": delay_bucket(q),
                        "evidence_location": evidence_location(view, window),
                        "question": q.query,
                        "answer": q.answer,
                        "prediction": pred,
                        "evidence_spans": aux.get("evidence_time_spans", torch.empty(0, 2)).tolist() if aux else [],
                    }
                    row.update(retrieval_metrics(aux, committed, window, tol))
                    if pred is not None and q.answer:
                        row.update(answer_scores(pred, q.answer))
                    rows.append(row)
                    fout.write(json.dumps(row) + "\n")
            fout.flush()
            if trace:
                os.makedirs(args.trace_dir, exist_ok=True)
                final = model.lvm_short_former.state_view(state[0])
                with open(os.path.join(args.trace_dir, f"eval_{args.track}_{item.stream_id}.json"), "w") as f:
                    json.dump({"clip_uid": item.stream_id, "stream_id": item.stream_id, "clip_uids": item.clip_uids, "arm": args.arm,
                               "track": args.track, "config": cfg_dict, "windows": windows, "frames": trace,
                               "queries": [r for r in rows if r["stream_id"] == item.stream_id],
                               "final_nodes": torch.stack([final["active_time_starts"], final["active_time_ends"]], -1).tolist()}, f)
            print(f"[eval] stream {ci + 1}/{len(ds)} {item.stream_id} clips={len(item.pieces)} frames={n_frames} queries={len(item.queries)} {time.time() - t0:.0f}s", flush=True)
    keys = ["recall_at_m", "t_overlap", "evidence_precision", "evidence_span_s", "evidence_survival", "best_iou", "num_committed", "exact", "contains"]
    groups = {"overall": rows}
    for name, fn in (
        ("delay", lambda r: r["delay_bucket"]),
        ("evidence", lambda r: r["evidence_location"]),
        ("linked", lambda r: "linked" if r["linked"] else "unlinked"),
        ("participant", lambda r: "disjoint" if r["participant_disjoint"] else "seen_in_train"),
        ("template", lambda r: r["template"] or "none"),
    ):
        buckets = defaultdict(list)
        for r in rows:
            buckets[fn(r)].append(r)
        for b, rs in buckets.items():
            groups[f"{name}={b}"] = rs
    summary = {
        "arm": args.arm,
        "track": args.track,
        "split": args.split,
        "ckpt": args.ckpt,
        "streams": len(ds),
        "read": {"ablation": args.read_ablation, "evidence_k": args.evidence_k, "open_node": not args.no_open, "blank_frame": args.blank_frame,
                 "frames_baseline": args.frames_baseline, "num_frames": args.num_frames},
        "groups": {g: summarize(rs, keys) for g, rs in groups.items()},
        "config": cfg_dict,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["groups"]["overall"], indent=1))
if __name__ == "__main__":
    main()
