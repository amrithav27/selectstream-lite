"""Train SelectStream on Ego4D NLQ + QaEgo4D, one clip per item, strictly online (docs/plan.md).

Stage R: arm A, trains reader + writer.   Stage W: one arm, trains the writer only.
The Q1 scorer is trained on its own in Stage 1 (streamlite/cli/train_scorer.py) and loaded with --scorer.
Each clip is streamed once at 1 fps; every query is read at its own t_q; the summed loss is
backpropagated once per clip. Resumes from <output_dir>/latest/ if present.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
import numpy as np
import torch
from streamlite.cli.common import load_yaml
from streamlite.cli.ego4d_common import (
    apply_arm,
    atomic_save,
    format_prompt,
    frame_labels,
    load_model,
    load_trainable_state,
    log_device,
    prepare_scorer,
    trace_extras,
    rng_state,
    scorer_pos_weight,
    set_rng_state,
    set_trainable,
    trainable_state_dict,
)
from streamlite.data.ego4d_qa import Ego4DStreamDataset
from streamlite.data.frame_cache import FrameCache, as_image, stream_inputs
from streamlite.model.importance_scorer import scorer_bce
from streamlite.trainer.method_sft import selectstream_query_loss
from streamlite.utils.misc import set_seed
from streamlite.utils.video import iter_stream_frames
STAGE_GROUPS = {"R": ("reader", "writer"), "W": ("writer",)}
def with_last(it):
    """Yield (*item, is_last) with one item of lookahead (the stream length is only known by decoding)."""
    it = iter(it)
    try:
        prev = next(it)
    except StopIteration:
        return
    for cur in it:
        yield (*prev, False)
        prev = cur
    yield (*prev, True)
def stream_frames(item, sample_fps: float, max_side: int):
    """(img, t, cut, last): `cut` closes the open segment after this frame, at the stream end and
    wherever the stream jumps over unshown video time (gap between two clips of the video)."""
    gap = 1.5 / sample_fps
    prev = None
    for cur in iter_stream_frames(item.pieces, sample_fps=sample_fps, max_side=max_side):
        if prev is not None:
            yield prev[0], prev[1], cur[1] - prev[1] > gap, False
        prev = cur
    if prev is not None:
        yield prev[0], prev[1], True, True
def save_checkpoint(path, model, opt, scaler, epoch, clip_pos, order, step, cfg_dict, args):
    atomic_save(
        {
            "model": trainable_state_dict(model),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "clip_pos": clip_pos,
            "order": order,
            "step": step,
            "rng": rng_state(),
            "config": cfg_dict,
            "args": vars(args),
        },
        os.path.join(path, "trainable.pt"),
    )
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego4d_qwen25vl7b.yaml")
    ap.add_argument("--torch_dtype", default=None, help="Override model.torch_dtype (bfloat16 | float16 | float32).")
    ap.add_argument("--stage", choices=sorted(STAGE_GROUPS), required=True)
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], default="A")
    ap.add_argument("--init_from", default=None, help="Checkpoint dir/file to start from (Stage W: the Stage-R checkpoint).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--clean_dir", default="dataset/clean")
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--part", choices=["train", "all"], default="train", help="train = train minus dev participants.")
    ap.add_argument("--max_clips", type=int, default=None, help="Use only the first N streams (pilot).")
    ap.add_argument("--clips_file", default=None, help="Keep only streams containing a listed clip_uid / video_uid (one per line), in file order.")
    ap.add_argument("--max_steps", type=int, default=None, help="Stop after N optimizer steps (Stage W budget).")
    ap.add_argument("--epochs", type=int, default=None, help="Override training.epochs (pilot).")
    ap.add_argument("--save_every_clips", type=int, default=25)
    ap.add_argument("--trace_dir", default=None, help="Write per-frame traces of the first --trace_clips clips for plots.")
    ap.add_argument("--trace_clips", type=int, default=3)
    ap.add_argument("--calibration", default="outputs/calibration.json", help="calibrate_write output; overrides the config arms block for B/C/D.")
    ap.add_argument("--feature_cache", default=None, help="Stage-0 cache dir (streamlite/cli/cache_features.py): write path reads g_t, queries read the cached frame.")
    ap.add_argument("--scorer", default=None, help="Stage-1 scorer (train_scorer.py); required for arm B.")
    ap.add_argument("--seed", type=int, default=None, help="Override training.seed in config (for multi-seed training P4).")
    ap.add_argument("--segment_center", choices=("none", "train_mean"), default="none",
                    help="Fresh runs only (a checkpoint carries its own): subtract the mean g_t of the training streams "
                         "before the segment encoder (none = the paper). Needs --feature_cache.")
    args = ap.parse_args()
    if args.stage == "R" and args.arm != "A":
        raise ValueError("Stage R trains the shared reader on the paper's arm A (docs/plan.md §3 training order).")
    cfg_dict = apply_arm(load_yaml(args.config), args.arm, args.calibration)
    if args.torch_dtype:
        cfg_dict["model"]["torch_dtype"] = args.torch_dtype
    if args.epochs is not None:
        cfg_dict["training"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg_dict["training"]["seed"] = int(args.seed)
    tcfg = cfg_dict["training"]
    dcfg = cfg_dict["data"]
    set_seed(int(tcfg.get("seed", 42)))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=1)  # the effective config, read by plot_ego4d for parameter boxes
    log_device()
    latest = os.path.join(args.output_dir, "latest", "trainable.pt")
    resume = torch.load(latest, map_location="cpu", weights_only=False) if os.path.exists(latest) else None
    model = load_model(cfg_dict)
    if resume is not None:
        load_trainable_state(model, latest)
    elif args.init_from:
        load_trainable_state(model, args.init_from)
    prepare_scorer(model, args.arm, args.scorer)
    cache = FrameCache(args.feature_cache, args.video_dir, float(dcfg.get("sample_fps", 1.0)), int(dcfg.get("frame_max_side", 448))) if args.feature_cache else None
    params = set_trainable(model, STAGE_GROUPS[args.stage])
    n_train = sum(p.numel() for p in params)
    print(f"[train] stage={args.stage} arm={args.arm} write_signal={model.cfg.lvm_write_signal} trainable={n_train/1e6:.1f}M params")
    opt = torch.optim.AdamW(params, lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.0)))
    backbone_fp16 = next(model.base_model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=backbone_fp16)
    sample_fps = float(dcfg.get("sample_fps", 1.0))
    ds = Ego4DStreamDataset(
        clean_dir=args.clean_dir,
        video_dir=args.video_dir,
        split="train",
        source="merged",
        part=args.part,
        dev_frac=float(dcfg.get("dev_frac", 0.1)),
        query_time_mode=str(dcfg.get("query_time_mode", "uniform")),
        sample_fps=sample_fps,
        resample_ask_times=True,
        seed=int(tcfg.get("seed", 42)),
        require_video=cache is None,
    )
    if cache is not None:
        n0 = len(ds.items)
        ds.items = [i for i in ds.items if all(cache.has(u) for u in i.clip_uids)]
        print(f"[data] feature cache {args.feature_cache}: {len(ds.items)}/{n0} streams fully cached")
    if args.segment_center == "train_mean" and resume is None and not args.init_from:
        if cache is None:
            raise ValueError("--segment_center train_mean needs --feature_cache")
        center = torch.from_numpy(cache.feature_mean(sorted({u for i in ds.items for u in i.clip_uids})))
        model.lvm_short_former.set_segment_center(center)
        np.save(os.path.join(args.output_dir, "segment_center.npy"), center.numpy())
        print(f"[train] segment encoder input centred on the mean g_t of {len(ds.items)} training streams (|mean| = {float(center.norm()):.2f})")
    if args.clips_file:
        # Keep the streams that contain a listed clip_uid (or are a listed video_uid), in file order.
        wanted = [l.strip() for l in open(args.clips_file) if l.strip()]
        by_uid = {u: i for i in ds.items for u in [i.stream_id, *i.clip_uids]}
        seen, kept = set(), []
        for u in wanted:
            it = by_uid.get(u)
            if it is not None and it.stream_id not in seen:
                seen.add(it.stream_id)
                kept.append(it)
        ds.items = kept
    if args.max_clips is not None:
        ds.items = ds.items[: args.max_clips]
    tol = float(tcfg.get("time_tolerance", 0.5))
    mean_q = float(np.mean([len(i.queries) for i in ds.items]))
    pos_weight = scorer_pos_weight(ds.items, tol)
    print(f"[data] streams={len(ds)} clips={sum(len(i.pieces) for i in ds.items)} frames={sum(i.num_frames for i in ds.items)} "
          f"queries={sum(len(i.queries) for i in ds.items)} mean_q/stream={mean_q:.2f} scorer_pos_weight={pos_weight:.2f}")
    beta, gamma = float(tcfg["beta_ret"]), float(tcfg["gamma_spar"])
    lam_scorer = float(tcfg.get("scorer_weight", 1.0)) if "scorer" in STAGE_GROUPS[args.stage] else 0.0
    grad_accum = int(tcfg.get("grad_accum", 4))
    epochs = int(tcfg.get("epochs", 1))
    oracle = model.cfg.lvm_write_signal == "oracle"
    start_epoch, start_pos, step = 0, 0, 0
    if resume is not None:
        opt.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        set_rng_state(resume["rng"])
        start_epoch, start_pos, step = resume["epoch"], resume["clip_pos"], resume["step"]
        print(f"[resume] epoch={start_epoch} stream_pos={start_pos} step={step}")
    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    model.train()
    model.base_model.eval()  # frozen backbone: no dropout, inference behaviour
    for epoch in range(start_epoch, epochs):
        ds.set_epoch(epoch)
        order = random.Random(epoch + int(tcfg.get("seed", 42))).sample(range(len(ds)), len(ds))
        if resume is not None and epoch == start_epoch and resume.get("order") is not None:
            order = resume["order"]
        pos0 = start_pos if epoch == start_epoch else 0
        opt.zero_grad(set_to_none=True)
        accum = 0
        for pos in range(pos0, len(order)):
            item = ds[order[pos]]
            t0 = time.time()
            state = model.init_memory_state(batch_size=1, dtype=model.trainable_dtype, device=model.trainable_device)
            windows = [(q.window_start, q.window_end) for q in item.queries]
            pending = list(item.queries)
            q_loss_sum, q_stats, logits, ys, trace, writes = 0.0, [], [], [], [], []
            n_frames, n_merge, n_nonfinite = 0, 0, 0
            for img, g, t, cut, last in stream_inputs(item, sample_fps, int(dcfg.get("frame_max_side", 448)), cache):
                n_frames += 1
                y_t = float(frame_labels([t], windows, tol)[0])
                state, info = model.update_memory_frame(img, t, state, flush=cut, frame_signal=y_t if oracle else None, g=g)
                if info.get("importance_logit") is not None:
                    logits.append(info["importance_logit"])
                    ys.append(y_t)
                if info.get("write") is not None:
                    writes.append(info["write"])
                n_merge += int(info.get("num_merge", 0))
                if args.trace_dir and pos < args.trace_clips:
                    trace.append({k: (v.tolist() if torch.is_tensor(v) else v) for k, v in info.items() if k != "importance_logit"}
                                 | {"importance": float(torch.sigmoid(info["importance_logit"].detach())) if info.get("importance_logit") is not None else None}
                                 | trace_extras(model, state, img, g, n_frames - 1))
                while pending and (pending[0].ask_time <= t + 1e-6 or last):
                    q = pending.pop(0)
                    loss_q, st = selectstream_query_loss(
                        model, as_image(img), format_prompt(q.query), q.answer, (q.window_start, q.window_end), state, beta, gamma, tol, t_now=t
                    )
                    # Backward per query: frees this query's activations now instead of holding every
                    # query's graph until the end of the stream. The write path is shared by later
                    # queries, so it is retained. Same gradient as summing the losses first.
                    # Equal weight per query (the paper's per-example loss), divided by the mean
                    # queries per stream so one stream's gradient is on the scale of one example.
                    # A non-finite loss would put NaN into the accumulated grads, and GradScaler then
                    # skips the whole 4-stream step; repeated, the weights never move. Drop the query instead.
                    if not torch.isfinite(loss_q.detach()):
                        n_nonfinite += 1
                        print(f"[train][WARN] non-finite loss on {q.query_id}: {st}", flush=True)
                    elif loss_q.requires_grad:
                        scaler.scale(loss_q / (mean_q * grad_accum)).backward(retain_graph=True)
                    if torch.isfinite(loss_q.detach()):
                        q_loss_sum += float(loss_q.detach())
                    del loss_q
                    st.update({"query_id": q.query_id, "ask_time": q.ask_time, "delay": q.ask_time - q.window_end})
                    q_stats.append(st)
            loss_scorer = torch.zeros((), device=model.trainable_device)
            if lam_scorer > 0 and logits:
                loss_scorer = scorer_bce(torch.stack(logits), torch.tensor(ys, device=model.trainable_device), pos_weight)
                scaler.scale(lam_scorer * loss_scorer / grad_accum).backward()
            loss = q_loss_sum / mean_q + lam_scorer * float(loss_scorer.detach())
            del state, logits
            accum += 1
            grad_norm, step_skipped = None, None
            if accum % grad_accum == 0:
                scaler.unscale_(opt)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("max_grad_norm", 1.0))))
                scale_before = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(opt)
                scaler.update()
                # GradScaler lowers the scale exactly when it skipped the step (inf/NaN grads).
                step_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
                opt.zero_grad(set_to_none=True)
                step += 1
            sims = [w["best_sim"] for w in writes if w.get("best_sim") is not None]
            rec = {
                "epoch": epoch,
                "pos": pos,
                "step": step,
                "stream_id": item.stream_id,
                "clips": len(item.pieces),
                "frames": n_frames,
                "queries": len(q_stats),
                "loss": loss,
                "loss_scorer": float(loss_scorer.detach()),
                "writes_new": sum(w["action"] == "new" for w in writes),
                "writes_update": sum(w["action"] == "update" for w in writes),
                "merges": n_merge,
                "nearest_cos_mean": float(np.mean(sims)) if sims else None,
                "nearest_cos_max": float(np.max(sims)) if sims else None,
                "evidence_location": {k: sum(s["evidence_location"] == k for s in q_stats) for k in ("open", "split", "committed")},
                "sec": round(time.time() - t0, 1),
                "grad_scale": float(scaler.get_scale()) if scaler.is_enabled() else None,  # fp16: a drop = skipped (overflowed) step
                "grad_norm": grad_norm,  # pre-clip norm, only on stream that ends a step
                "step_skipped": step_skipped,
                "nonfinite_queries": n_nonfinite,
                "max_mem_gb": [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())],
            }
            for key in ("loss_ans", "loss_ret", "loss_spar", "recall_at_m", "t_overlap", "evidence_survival", "num_committed",
                        "evidence_span_s", "evidence_precision", "loss_ret_coarse", "loss_ret_chance", "ret_logit_scale", "prior_weight", "bias_eta"):
                vals = [s[key] for s in q_stats if s.get(key) is not None]
                rec[key] = float(np.mean(vals)) if vals else None
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if trace:
                os.makedirs(args.trace_dir, exist_ok=True)
                with open(os.path.join(args.trace_dir, f"train_{item.stream_id}.json"), "w") as f:
                    json.dump({"clip_uid": item.stream_id, "stream_id": item.stream_id, "clip_uids": item.clip_uids, "arm": args.arm,
                               "stage": args.stage, "config": cfg_dict, "windows": windows, "queries": q_stats, "frames": trace}, f)
            if (pos + 1) % args.save_every_clips == 0:
                save_checkpoint(os.path.join(args.output_dir, "latest"), model, opt, scaler, epoch, pos + 1, order, step, cfg_dict, args)
            if args.max_steps is not None and step >= args.max_steps:
                break
        if accum % grad_accum != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("max_grad_norm", 1.0)))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
        save_checkpoint(os.path.join(args.output_dir, f"epoch{epoch}"), model, opt, scaler, epoch + 1, 0, None, step, cfg_dict, args)
        atomic_save({"model": trainable_state_dict(model), "config": cfg_dict, "step": step}, os.path.join(args.output_dir, f"epoch{epoch}", "weights.pt"))
        save_checkpoint(os.path.join(args.output_dir, "latest"), model, opt, scaler, epoch + 1, 0, None, step, cfg_dict, args)
        resume = None
        if args.max_steps is not None and step >= args.max_steps:
            break
    # Fixed path for downstream jobs: the last weights, whatever epoch max_steps stopped in.
    atomic_save({"model": trainable_state_dict(model), "config": cfg_dict, "step": step}, os.path.join(args.output_dir, "final", "weights.pt"))
    print("[train] done")
if __name__ == "__main__":
    main()
