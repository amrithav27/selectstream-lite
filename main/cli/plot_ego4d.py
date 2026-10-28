"""Figures for the Ego4D SelectStream runs (style: main/cli/plot_style.py).

Every figure: labelled axes with units, a legend, and a grouped parameter footer; feature maps and the
memory are animated (GIF via Pillow; no ffmpeg on Ada). Uncertainty: bootstrap 95% CIs over videos.

    python -m main.cli.plot_ego4d --run_dir outputs/full                 # everything a run has produced
    python -m main.cli.plot_ego4d --traces <dir> --out <figs>            # per-stream figures
    python -m main.cli.plot_ego4d --train_log <dir>/train_log.jsonl --out <figs>
    python -m main.cli.plot_ego4d --summaries <a>/summary.json ... --out <figs>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from main.cli.plot_style import (ARM_COLOR, ARM_LABEL, ARM_ORDER, AXIS, GRID, INK, INK_2, MUTED, Q2_COLOR, SERIES, SIGNAL_COLOR,
                                 ARM_TICK, arm_key, bootstrap_mean, cfg_groups, fig_legend_y, footer, paired_diff, plt, save, suptitle, trailing_mean)

DELAY_ORDER = ["[0,16)s", "[16,64)s", "[64,256)s", "[256,1024)s", "[1024,inf)s"]
EVAL_METRICS = [("best_iou", "best-node IoU at t_q", "write side: is the evidence kept, tightly"),
                ("evidence_survival", "evidence survival", "any committed node overlaps the window"),
                ("recall_at_m", "Recall@M", "a retrieved node overlaps the window (Eq. 56)"),
                ("t_overlap", "T-Overlap", "coverage of the window by retrieved spans (Eq. 57)"),
                ("evidence_precision", "evidence precision", "retrieved seconds inside the window"),
                ("contains", "answer contains-match", "QA track only")]


def read_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def gap(t, y, max_dt=1.5):
    """Break a line where the stream jumps over unshown video time (gap between clips)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    cut = np.where(np.diff(t) > max_dt)[0] + 1
    return np.insert(t, cut, np.nan), np.insert(y, cut, np.nan)


def shade_windows(ax, windows, label=True):
    for i, (s, e) in enumerate(windows):
        ax.axvspan(s, max(e, s + 0.5), color=SIGNAL_COLOR["window"], alpha=0.16, lw=0, label="answer window (GT)" if (label and i == 0) else None)


def ask_ticks(ax, queries, label=True):
    for i, q in enumerate(queries):
        ax.axvline(q["ask_time"], ymin=0.0, ymax=0.06, color=SIGNAL_COLOR["ask"], lw=1.4, label="question asked (t_q)" if (label and i == 0) else None)


# ----------------------------------------------------------------------------------------------- per stream
def plot_stream(tr: dict, out: str) -> None:
    fr, cfg = tr["frames"], tr["config"]
    t = np.array([f["t"] for f in fr])
    has_imp = any(f.get("importance") is not None for f in fr)
    signal = cfg["vismem"].get("lvm_write_signal")
    rows = 3 if has_imp else 2
    fig, axes = plt.subplots(rows, 1, figsize=(13, 2.7 * rows + 2.6), sharex=True, gridspec_kw={"height_ratios": [3] + [1.6] * (rows - 1), "hspace": 0.28})
    ax = axes[0]
    shade_windows(ax, tr["windows"])
    ax.plot(*gap(t, [f["s_feat"] for f in fr]), color=SIGNAL_COLOR["s"], lw=0.8, label="raw surprise $s_t = 1-\\cos(g_t, g_{t-1})$")
    ax.plot(*gap(t, [f["s_ema"] for f in fr]), color=SIGNAL_COLOR["sbar"], lw=1.8, label="surprise $\\bar s_t$ (EMA, Eq. 2)")
    if signal != "surprise":
        ax.plot(*gap(t, [f["w"] for f in fr]), color=SIGNAL_COLOR["w"], lw=1.8, label=f"write signal $w_t$ ({signal})")
    ax.plot(*gap(t, [f["theta_high"] for f in fr]), color=SIGNAL_COLOR["theta"], lw=1.0, ls=(0, (4, 3)), label="cut threshold $\\theta_t$ (recent quantile)")
    for i, c in enumerate([f["t"] for f in fr if f["cut"]]):
        ax.axvline(c, color=SIGNAL_COLOR["cut"], lw=0.6, alpha=0.7, label="segment cut (Eq. 3)" if i == 0 else None, zorder=0)
    ask_ticks(ax, tr.get("queries", []))
    ax.set_ylabel("signal (unitless)")
    ax.set_title("Write-path signals and cuts")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5)

    ax = axes[1]
    ax.step(*gap(t, [f.get("num_active", 0) for f in fr]), where="post", color=SIGNAL_COLOR["node"], lw=1.6, label="committed nodes $|M_t|$")
    ax.axhline(cfg["vismem"].get("lvm_num_slots"), color=MUTED, ls=(0, (4, 3)), lw=1.0, label="capacity N")
    merges = [f["t"] for f in fr if f.get("num_merge")]
    ax.plot(merges, [cfg["vismem"].get("lvm_num_slots")] * len(merges), lw=0, marker="v", ms=5, color=SERIES[7], label="merge (Eq. 5)")
    ax.set_ylabel("nodes (count)")
    ax.set_title("Memory size")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5)

    if has_imp:
        ax = axes[2]
        shade_windows(ax, tr["windows"], label=False)
        ax.plot(*gap(t, [f.get("importance") for f in fr]), color=SIGNAL_COLOR["a"], lw=1.1, label="scorer $a_t=\\sigma(\\mathrm{MLP}(\\phi_t))$")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("P(asked about)")
        ax.set_title("Learned importance (Q1 scorer)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5)
    axes[-1].set_xlabel("video time (s) · line breaks = video between clips that is not shown")
    sid = tr.get("stream_id", tr.get("clip_uid", "?"))
    suptitle(fig, f"Stream {sid[:8]} · arm {tr.get('arm')} · {tr.get('stage') or tr.get('track') or ''}",
             f"{len(tr.get('clip_uids', [1]))} clip(s), {len(fr)} frames, {len(tr.get('queries', []))} questions, "
             f"{sum(f['cut'] for f in fr)} cuts, {sum(int(f.get('num_merge', 0)) for f in fr)} merges")
    footer(fig, cfg_groups(cfg, {"Data": {"stream": sid[:8]}}), reserve_in=0.6)
    fig.subplots_adjust(right=0.80)
    save(fig, out)


def plot_nodes(tr: dict, out: str) -> None:
    if "final_nodes" not in tr:
        return
    nodes = sorted(tr["final_nodes"], key=lambda s: s[0])
    fig, ax = plt.subplots(figsize=(13, 5.2))
    shade_windows(ax, tr["windows"])
    for i, (s, e) in enumerate(nodes):
        ax.barh(i, max(e - s, 0.8), left=s, height=0.62, color=SIGNAL_COLOR["node"], label="node span $[t_{start}, t_{end}]$" if i == 0 else None)
    ask_ticks(ax, tr.get("queries", []))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("node (sorted by start)")
    ax.set_yticks(range(len(nodes)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.5)
    sid = tr.get("stream_id", tr.get("clip_uid", "?"))
    suptitle(fig, f"Memory at the end of stream {sid[:8]} · arm {tr.get('arm')}",
             f"{len(nodes)} committed nodes; wide bars are merged or updated nodes")
    footer(fig, cfg_groups(tr["config"], {"Data": {"stream": sid[:8]}}), reserve_in=0.55)
    fig.subplots_adjust(right=0.80)
    save(fig, out)


def animate_memory(tr: dict, out: str, stride: int = 4) -> None:
    fr = [f for f in tr["frames"] if f.get("node_spans") is not None]
    if len(fr) < 3:
        return
    fr = fr[::stride]
    t_end = fr[-1]["t"]
    n_max = max(len(f["node_spans"]) for f in fr)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14)

    def draw(i):
        ax.clear()
        f = fr[i]
        shade_windows(ax, [w for w in tr["windows"] if w[0] <= f["t"]])
        for j, (s, e) in enumerate(sorted(f["node_spans"], key=lambda x: x[0])):
            ax.barh(j, max(e - s, 0.8), left=s, height=0.62, color=SIGNAL_COLOR["node"], label="node span" if j == 0 else None)
        ax.axvline(f["t"], color=SIGNAL_COLOR["ask"], lw=1.2, label="now")
        ax.set_xlim(fr[0]["t"], t_end)
        ax.set_ylim(-0.8, max(n_max, 1) - 0.2)
        ax.set_xlabel("video time (s)")
        ax.set_ylabel("node (sorted by start)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"Memory over time · arm {tr.get('arm')} · t = {f['t']:.0f} s · {len(f['node_spans'])} nodes (N = {tr['config']['vismem'].get('lvm_num_slots')})",
                     fontsize=10.5)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(fr), interval=120, blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=8))
    plt.close(fig)


def animate_feature_map(tr: dict, out: str, stride: int = 2) -> None:
    fr = [f for f in tr["frames"] if f.get("token_norms")][::stride]
    if len(fr) < 3:
        return
    h, w = fr[0]["grid"]
    fr = [f for f in fr if f.get("grid") == fr[0]["grid"]]
    maps = [np.asarray(f["token_norms"], dtype=float)[: h * w].reshape(h, w) for f in fr]
    vmin, vmax = np.percentile(np.stack(maps), [2, 98])
    fig, ax = plt.subplots(figsize=(7, 4.6))
    fig.subplots_adjust(left=0.06, right=0.88, top=0.86, bottom=0.12)
    im = ax.imshow(maps[0], cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")
    ax.grid(False)
    ax.set_xlabel(f"merged-patch column (28 px each, {w} total)")
    ax.set_ylabel(f"merged-patch row ({h} total)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("‖Φ_v token‖ (L2 norm, unitless)")
    ttl = ax.set_title("")
    ax.legend(handles=[Patch(color="#2a78d6", label="high-norm visual token")], loc="upper right", fontsize=8)

    def draw(i):
        im.set_data(maps[i])
        ttl.set_text(f"Visual-token norms · stream {tr.get('stream_id', '?')[:8]} · t = {fr[i]['t']:.0f} s")
        return [im, ttl]

    anim = animation.FuncAnimation(fig, draw, frames=len(maps), interval=150, blit=False)
    anim.save(out, writer=animation.PillowWriter(fps=6))
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- training
def _run_cfg(path: str, cfg_path: Optional[str] = None) -> dict:
    run_cfg = os.path.join(os.path.dirname(path), "config.json")
    if os.path.exists(run_cfg):
        return json.load(open(run_cfg))
    if cfg_path and os.path.exists(cfg_path):
        import yaml

        return yaml.safe_load(open(cfg_path))
    return {}


def _series(ax, x, vals, color, label, k):
    v = np.array([np.nan if a is None else a for a in vals], dtype=float)
    if np.all(np.isnan(v)):
        return False
    ax.plot(x, v, lw=0, marker="o", ms=2.4, alpha=0.25, color=color)
    ax.plot(x, trailing_mean(vals, k), color=color, lw=1.8, label=label)
    return True


def plot_train_log(path: str, out_dir: str, cfg_path: Optional[str] = None, name: Optional[str] = None) -> None:
    rows = read_jsonl(path)
    if not rows:
        return
    cfg = _run_cfg(path, cfg_path)
    name = name or os.path.basename(os.path.dirname(path))
    x = np.arange(1, len(rows) + 1)
    k = max(1, min(25, len(rows) // 6))
    arm = cfg.get("arm_values", {}).get("arm", "?")
    steps = rows[-1].get("step")
    extra = {"Training": {"stage": name, "streams seen": len(rows), "optimizer steps": steps, "smoothing": f"trailing {k}-stream mean"}}
    sub = f"{len(rows)} streams, {steps} optimizer steps · dots = single streams, lines = trailing {k}-stream mean"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.18)
    panels = [(axes[0, 0], [("loss_ans", "$\\mathcal{L}_{ans}$ (answer NLL)")], "nats per token", "Answer loss"),
              (axes[0, 1], [("loss_ret", "$\\mathcal{L}_{ret}$ (Eq. 36)")], "nats", "Retrieval loss"),
              (axes[1, 0], [("loss_spar", "$\\mathcal{L}_{spar}$ (Eq. 37–38)")], "unitless", "Sparsity loss"),
              (axes[1, 1], [("recall_at_m", "Recall@M"), ("t_overlap", "T-Overlap"), ("evidence_precision", "evidence precision")],
               "fraction of questions", "Retrieval on the training stream")]
    for ax, keys, ylab, title in panels:
        for i, (key, lab) in enumerate(keys):
            _series(ax, x, [r.get(key) for r in rows], SERIES[i] if len(keys) > 1 else ARM_COLOR.get(arm, SERIES[0]), lab, k)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8.5)
    axes[1, 1].set_ylim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("training streams processed (count)")
    suptitle(fig, f"Training {name} · arm {arm}", sub)
    footer(fig, cfg_groups(cfg, extra), reserve_in=0.55)
    save(fig, os.path.join(out_dir, f"train_{name}_learning.png"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    fig.subplots_adjust(hspace=0.32, wspace=0.2)
    ax = axes[0, 0]
    gx = [i for i, r in zip(x, rows) if r.get("grad_norm") is not None]
    ax.plot(gx, [r["grad_norm"] for r in rows if r.get("grad_norm") is not None], marker="o", ms=3, color=SERIES[0], label="gradient norm before clipping")
    ax.axhline(cfg.get("training", {}).get("max_grad_norm", 1.0), color=MUTED, ls=(0, (4, 3)), lw=1, label="clip threshold")
    skipped = [i for i, r in zip(x, rows) if r.get("step_skipped")]
    if skipped:
        ax.plot(skipped, [ax.get_ylim()[1]] * len(skipped), lw=0, marker="x", color=SERIES[7], label="step skipped (fp16 overflow)")
    ax.set_yscale("log")
    ax.set_ylabel("L2 norm (log scale)")
    ax.set_title("Gradient health")
    ax.legend(loc="best", fontsize=8.5)
    ax = axes[0, 1]
    fps = [r["frames"] / r["sec"] if r.get("sec") else None for r in rows]
    qps = [r["sec"] / r["queries"] if r.get("queries") else None for r in rows]
    _series(ax, x, fps, SERIES[0], "frames per second", k)
    ax.set_ylabel("frames / s")
    ax.set_title(f"Throughput · median {np.nanmedian([np.nan if v is None else v for v in qps]):.1f} s / question")
    ax.legend(loc="best", fontsize=8.5)
    ax = axes[1, 0]
    for i, (key, lab) in enumerate((("writes_new", "new node"), ("writes_update", "update existing node"), ("merges", "merge"))):
        _series(ax, x, [r.get(key) for r in rows], SERIES[i], lab, k)
    ax.set_ylabel("events per stream (count)")
    ax.set_title("Memory writes")
    ax.legend(loc="best", fontsize=8.5)
    ax = axes[1, 1]
    mem = np.array([r.get("max_mem_gb") or [] for r in rows], dtype=object)
    n_dev = max((len(m) for m in mem), default=0)
    for d in range(n_dev):
        ax.plot(x, [m[d] if len(m) > d else np.nan for m in mem], color=SERIES[d], lw=1.4, label=f"GPU {d}" + (" (trainables)" if d == 0 else " (backbone)"))
    ax.axhline(11.0, color=SERIES[7], ls=(0, (4, 3)), lw=1, label="card memory (11 GB)")
    ax.set_ylabel("peak allocated (GB)")
    ax.set_title("GPU memory")
    ax.legend(loc="best", fontsize=8, ncol=2)
    for a in axes[1]:
        a.set_xlabel("training streams processed (count)")
    suptitle(fig, f"Training health {name} · arm {arm}", sub)
    footer(fig, cfg_groups(cfg, extra), reserve_in=0.55)
    save(fig, os.path.join(out_dir, f"train_{name}_health.png"))


def plot_stage_w_arms(run_dir: str, out_dir: str) -> None:
    logs = {arm_key(os.path.basename(os.path.dirname(p))): p for p in sorted(glob.glob(os.path.join(run_dir, "W_*", "train_log.jsonl")))}
    if len(logs) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.2)
    cfg = {}
    for arm in [a for a in ARM_ORDER if a in logs]:
        rows = read_jsonl(logs[arm])
        cfg = _run_cfg(logs[arm])
        x = np.arange(1, len(rows) + 1)
        k = max(1, min(25, len(rows) // 6))
        for ax, key in zip(axes, ("loss_ret", "loss_ans")):
            ax.plot(x, trailing_mean([r.get(key) for r in rows], k), color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    axes[0].set_ylabel("$\\mathcal{L}_{ret}$ (nats)")
    axes[0].set_title("Retrieval loss, writer retrained per arm")
    axes[1].set_ylabel("$\\mathcal{L}_{ans}$ (nats per token)")
    axes[1].set_title("Answer loss, writer retrained per arm")
    for ax in axes:
        ax.set_xlabel("training streams processed (count)")
    axes[1].legend(loc="upper right")
    suptitle(fig, "Stage W: same streams, same steps, one writer per arm", "lines = trailing mean; reader frozen from Stage R")
    footer(fig, cfg_groups(cfg, {"Training": {"stage": "W", "arms": ", ".join(a for a in ARM_ORDER if a in logs)}}), reserve_in=0.55)
    save(fig, os.path.join(out_dir, "train_W_arms.png"))


# ----------------------------------------------------------------------------------------------- evaluation
def load_evals(run_dir: str, track: str) -> Dict[str, dict]:
    out = {}
    for d in sorted(glob.glob(os.path.join(run_dir, f"eval_{track}_*"))):
        name = os.path.basename(d)[len(f"eval_{track}_"):]
        if re.search(r"_N\d+$", name) or not os.path.exists(os.path.join(d, "predictions.jsonl")):
            continue
        out[arm_key(name)] = {"rows": read_jsonl(os.path.join(d, "predictions.jsonl")), "summary": json.load(open(os.path.join(d, "summary.json")))}
    return out


def arm_footer(evals: Dict[str, dict], extra: Optional[dict] = None) -> Dict[str, Dict]:
    any_s = next(iter(evals.values()))["summary"]
    groups = cfg_groups(any_s["config"], extra)
    for g in ("Write",):
        for key in ("signal", "B_s", "q/window"):
            groups[g].pop(key, None)
    groups["Memory"].pop("τ_r", None)
    groups["Memory"].pop("τ_s", None)
    groups["Arms"] = {}
    for arm, e in evals.items():
        av = e["summary"]["config"].get("arm_values", {})
        v = e["summary"]["config"]["vismem"]
        bits = [f"{v.get('lvm_write_signal')}"]
        if v.get("lvm_write_signal") == "fixed":
            bits.append(f"L={v.get('lvm_fixed_segment_len')}")
        else:
            bits += [f"B_s={v.get('lvm_saw_energy_budget'):.3g}", f"q={v.get('lvm_saw_quantile'):.3g}"]
        bits += [f"τ_s={v.get('lvm_tau_s'):.3g}", f"τ_r={v.get('lvm_tau_r'):.3g}", f"ckpt={os.path.basename(str(e['summary'].get('ckpt')).rstrip('/'))}"]
        groups["Arms"][arm] = "(" + ", ".join(bits) + ")"
    return groups


def plot_eval_overview(evals: Dict[str, dict], track: str, out_dir: str) -> None:
    arms = [a for a in ARM_ORDER if a in evals]
    metrics = [m for m in EVAL_METRICS if m[0] != "contains" or track == "qa"]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.6 * nrow + 2.4), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.18)
    for ax, (key, lab, note) in zip(axes.flat, metrics):
        stats = [bootstrap_mean(evals[a]["rows"], key) for a in arms]
        means = [s[0] if s else np.nan for s in stats]
        bars = ax.bar(range(len(arms)), means, width=0.66, color=[ARM_COLOR[a] for a in arms], edgecolor="#fcfcfb", linewidth=2)
        for i, s in enumerate(stats):
            if s:
                ax.plot([i, i], [s[1], s[2]], color=INK_2, lw=1.2)
                ax.text(i, s[2] + 0.015, f"{s[0]:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(range(len(arms)), [ARM_TICK.get(a, a) for a in arms], fontsize=8.5)
        ax.set_ylim(0, min(1.0, max([s[2] for s in stats if s] + [0.1]) * 1.25))
        ax.set_ylabel("mean over questions (0–1)")
        ax.set_title(lab, fontsize=10.5, pad=20)
        ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    handles = [Patch(color=ARM_COLOR[a], label=ARM_LABEL[a]) for a in arms] + [Line2D([], [], color=INK_2, lw=1.2, label="95% CI (bootstrap over videos)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, fig_legend_y(fig)), ncol=len(handles), fontsize=8.5)
    n = {a: len(evals[a]["rows"]) for a in arms}
    suptitle(fig, f"Evaluation · val {track.upper()} track · all arms", f"{next(iter(n.values()))} questions per arm, "
             f"{len({r['stream_id'] for r in evals[arms[0]]['rows']})} video streams; bars = mean, whiskers = 95% CI")
    footer(fig, arm_footer(evals, {"Data": {"split": "val", "track": track}}), reserve_in=0.35)
    save(fig, os.path.join(out_dir, f"eval_{track}_overview.png"), top_in=1.45)


def plot_eval_vs_a(evals: Dict[str, dict], track: str, out_dir: str) -> None:
    if "A" not in evals or len(evals) < 2:
        return
    arms = [a for a in ARM_ORDER if a in evals and a != "A"]
    metrics = [m for m in EVAL_METRICS if m[0] != "contains" or track == "qa"]
    fig, ax = plt.subplots(figsize=(12, 0.62 * len(metrics) * max(2, len(arms)) / 2 + 3.6))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    off = np.linspace(-0.28, 0.28, len(arms)) if len(arms) > 1 else [0.0]
    for j, arm in enumerate(arms):
        for i, (key, lab, _) in enumerate(metrics):
            d = paired_diff(evals["A"]["rows"], evals[arm]["rows"], key)
            if not d:
                continue
            y = len(metrics) - 1 - i - off[j]
            ax.plot([d[1], d[2]], [y, y], color=ARM_COLOR[arm], lw=2, solid_capstyle="round")
            ax.plot(d[0], y, "o", ms=8, color=ARM_COLOR[arm], mec="#fcfcfb", mew=1.5, label=ARM_LABEL[arm] if i == 0 else None)
            ax.text(d[2], y, f"  {d[0]:+.3f}", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_yticks(range(len(metrics)), [m[1] for m in metrics][::-1])
    ax.set_xlabel("difference from arm A (same questions, same ask times; metric units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0))
    suptitle(fig, f"Each arm minus surprise (arm A) · val {track.upper()} track",
             "dot = mean paired difference, bar = 95% CI from resampling videos; right of 0 = better than surprise")
    footer(fig, arm_footer(evals, {"Data": {"split": "val", "track": track}}), reserve_in=0.5)
    fig.subplots_adjust(left=0.17, right=0.80)
    save(fig, os.path.join(out_dir, f"eval_{track}_vs_A.png"))


def plot_eval_delay(evals: Dict[str, dict], track: str, out_dir: str) -> None:
    arms = [a for a in ARM_ORDER if a in evals]
    buckets = [b for b in DELAY_ORDER if any(r["delay_bucket"] == b for e in evals.values() for r in e["rows"])]
    if not buckets:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.subplots_adjust(wspace=0.2)
    for ax, (key, lab) in zip(axes, (("best_iou", "best-node IoU at t_q"), ("recall_at_m", "Recall@M"))):
        for arm in arms:
            st = [bootstrap_mean([r for r in evals[arm]["rows"] if r["delay_bucket"] == b], key) for b in buckets]
            m = [s[0] if s else np.nan for s in st]
            ax.fill_between(range(len(buckets)), [s[1] if s else np.nan for s in st], [s[2] if s else np.nan for s in st], color=ARM_COLOR[arm], alpha=0.12, lw=0)
            ax.plot(range(len(buckets)), m, marker="o", ms=6, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        counts = [sum(r["delay_bucket"] == b for r in evals[arms[0]]["rows"]) for b in buckets]
        ax.set_xticks(range(len(buckets)), [f"{b[:-1]}\nn={c}" for b, c in zip(buckets, counts)])
        ax.set_xlabel("delay between evidence end and question, t_q − t_end (s)")
        ax.set_ylabel(f"{lab} (0–1)")
        ax.set_title(lab)
    axes[1].legend(loc="upper right", fontsize=8.5)
    suptitle(fig, f"How long the evidence lasts · val {track.upper()} track", "line = mean, band = 95% CI over videos; n = questions per bucket")
    footer(fig, arm_footer(evals, {"Data": {"split": "val", "track": track}}), reserve_in=0.75)
    save(fig, os.path.join(out_dir, f"eval_{track}_delay.png"))


def plot_eval_slices(evals: Dict[str, dict], track: str, out_dir: str) -> None:
    arms = [a for a in ARM_ORDER if a in evals]
    slicers = [("question family", lambda r: (r.get("template") or "none").split(":")[0]),
               ("wearer seen in training?", lambda r: "unseen (disjoint)" if r.get("participant_disjoint") else "seen"),
               ("evidence location at t_q", lambda r: r.get("evidence_location") or "?")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    fig.subplots_adjust(hspace=0.62, wspace=0.18)
    width = 0.8 / len(arms)
    for col, (title, fn) in enumerate(slicers):
        cats = sorted({fn(r) for r in evals[arms[0]]["rows"]})
        for row, (key, lab) in enumerate((("best_iou", "best-node IoU"), ("recall_at_m", "Recall@M"))):
            ax = axes[row, col]
            for j, arm in enumerate(arms):
                st = [bootstrap_mean([r for r in evals[arm]["rows"] if fn(r) == c], key) for c in cats]
                xs = np.arange(len(cats)) + (j - (len(arms) - 1) / 2) * width
                ax.bar(xs, [s[0] if s else np.nan for s in st], width * 0.92, color=ARM_COLOR[arm], edgecolor="#fcfcfb", lw=1, label=ARM_LABEL[arm])
                for x0, s in zip(xs, st):
                    if s:
                        ax.plot([x0, x0], [s[1], s[2]], color=INK_2, lw=0.9)
            n = [sum(fn(r) == c for r in evals[arms[0]]["rows"]) for c in cats]
            ax.set_xticks(range(len(cats)), [f"{c}\nn={k}" for c, k in zip(cats, n)], fontsize=8.5)
            ax.set_ylabel(f"{lab} (0–1)")
            ax.set_title(f"{lab} by {title}", fontsize=10)
    handles = [Patch(color=ARM_COLOR[a], label=ARM_LABEL[a]) for a in arms] + [Line2D([], [], color=INK_2, lw=1, label="95% CI")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, fig_legend_y(fig)), ncol=len(handles), fontsize=8.5)
    suptitle(fig, f"Where the arms differ · val {track.upper()} track", "the unseen-wearer slice checks that a learned gain is not wearer recognition")
    footer(fig, arm_footer(evals, {"Data": {"split": "val", "track": track}}), reserve_in=0.45)
    save(fig, os.path.join(out_dir, f"eval_{track}_slices.png"), top_in=1.45)


def plot_budget(run_dir: str, out_dir: str) -> None:
    runs = {}
    for d in glob.glob(os.path.join(run_dir, "eval_qa_A_N*")):
        n = int(re.search(r"_N(\d+)$", d).group(1))
        if os.path.exists(os.path.join(d, "predictions.jsonl")):
            runs[n] = {"rows": read_jsonl(os.path.join(d, "predictions.jsonl")), "summary": json.load(open(os.path.join(d, "summary.json")))}
    runs = {n: r for n, r in runs.items() if r["rows"]}  # an N run with no stream long enough has no questions
    if len(runs) < 2:
        return
    Ns = sorted(runs)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for i, (key, lab) in enumerate((("best_iou", "best-node IoU"), ("evidence_survival", "evidence survival"), ("recall_at_m", "Recall@M"),
                                     ("contains", "answer contains-match"))):
        st = [bootstrap_mean(runs[n]["rows"], key) for n in Ns]
        ax.fill_between(Ns, [s[1] if s else np.nan for s in st], [s[2] if s else np.nan for s in st], color=SERIES[i], alpha=0.12, lw=0)
        ax.plot(Ns, [s[0] if s else np.nan for s in st], marker="o", ms=6, color=SERIES[i], label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns, [str(n) for n in Ns])
    ax.set_xlabel("memory budget N at evaluation (nodes; trained at 16)")
    ax.set_ylabel("mean over questions (0–1)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0))
    q = len(runs[Ns[0]]["rows"])
    suptitle(fig, "Memory budget · arm A · longest val streams", f"{q} questions on {len({r['stream_id'] for r in runs[Ns[0]]['rows']})} streams; band = 95% CI over videos")
    footer(fig, cfg_groups(runs[Ns[0]]["summary"]["config"], {"Memory": {"N": ", ".join(map(str, Ns)) + " (eval only)"}}), reserve_in=0.55)
    fig.subplots_adjust(right=0.78)
    save(fig, os.path.join(out_dir, "eval_budget_N.png"))


def plot_summaries(paths: List[str], out_dir: str) -> None:
    """Back-compatible single-run plot from summary.json files (no CIs: summaries carry means only)."""
    sums = [json.load(open(p)) for p in paths]
    buckets = [b for b in DELAY_ORDER if any(f"delay={b}" in s["groups"] for s in sums)]
    keys = [(k, lab) for k, lab, _ in EVAL_METRICS if any(s["groups"]["overall"].get(k) is not None for s in sums)]
    if not keys:
        print(f"[plot] summaries have no questions, nothing to plot: {paths}")
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys) + 1, 4.8), sharey=True, squeeze=False)
    fig.subplots_adjust(wspace=0.12)
    for ax, (key, lab) in zip(axes[0], keys):
        for i, s in enumerate(sums):
            arm = arm_key(str(s.get("arm")))
            ax.plot(range(len(buckets)), [s["groups"].get(f"delay={b}", {}).get(key) or np.nan for b in buckets], marker="o",
                    color=ARM_COLOR.get(arm, SERIES[i % 8]), label=f"arm {s['arm']} ({s['track']})")
        ax.set_xticks(range(len(buckets)), [b[:-1] for b in buckets], rotation=30, fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("delay t_q − t_end (s)")
    axes[0][0].set_ylabel("mean over questions (0–1)")
    axes[0][-1].legend(loc="upper right", fontsize=8)
    suptitle(fig, f"Evaluation by delay · {sums[0]['split']} · {sums[0]['track']}", f"{sums[0]['groups']['overall']['n']} questions")
    footer(fig, cfg_groups(sums[0]["config"]), reserve_in=0.8)
    save(fig, os.path.join(out_dir, "eval_by_delay.png"))


# ----------------------------------------------------------------------------------------------- checks and Q2
def plot_check1(run_dir: str, out_dir: str) -> None:
    path = os.path.join(run_dir, "scorer", "check1.json")
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    cfg = json.load(open(os.path.join(run_dir, "scorer", "config.json"))) if os.path.exists(os.path.join(run_dir, "scorer", "config.json")) else {}
    sel = next(r for r in res["runs"] if r["seed"] == res["selected"]["seed"] and not r.get("diagnostic"))
    scores = os.path.join(run_dir, "scorer", "check1_scores.npz")
    names = {"sbar": "surprise $\\bar s_t$ (arm A)", "s": "raw surprise $s_t$", "a": "scorer $a_t$", "w": "EMA($a_t$) (arm B's $w_t$)"}
    col = {"sbar": SIGNAL_COLOR["sbar"], "s": SIGNAL_COLOR["s"], "a": SIGNAL_COLOR["a"], "w": SIGNAL_COLOR["w"]}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    fig.subplots_adjust(wspace=0.22)
    ax = axes[0]
    if os.path.exists(scores):
        z = np.load(scores)
        y = z["val_nlq_y"]
        for k in ("sbar", "s", "a", "w"):
            x = z[f"val_nlq_{k}"]
            o = np.argsort(-x, kind="mergesort")
            tp = np.cumsum(y[o])
            ax.plot(tp / max(y.sum(), 1), tp / np.arange(1, len(y) + 1), color=col[k], lw=1.6,
                    label=f"{names[k]} · AP {sel['metrics']['val_nlq'][k]['ap']:.3f}")
        ax.axhline(y.mean(), color=MUTED, ls=(0, (4, 3)), lw=1, label=f"base rate {y.mean():.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    ax.set_xlabel("recall (fraction of answer-window frames found)")
    ax.set_ylabel("precision (fraction of flagged frames in a window)")
    ax.set_title("Precision–recall, val NLQ track")
    ax.legend(loc="upper right", fontsize=8.3)

    ax = axes[1]
    splits = [s for s in ("dev", "val_nlq", "val_qa", "val_nlq_disjoint", "val_qa_disjoint") if sel["metrics"].get(s)]
    width = 0.8 / 4
    for j, k in enumerate(("sbar", "s", "a", "w")):
        vals = [sel["metrics"][s][k]["ap"] for s in splits]
        xs = np.arange(len(splits)) + (j - 1.5) * width
        ax.bar(xs, vals, width * 0.92, color=col[k], edgecolor="#fcfcfb", lw=1, label=names[k])
        if k == "a":
            sd = [np.std([r["metrics"][s]["a"]["ap"] for r in res["runs"] if not r.get("diagnostic") and r["metrics"].get(s)]) for s in splits]
            ax.errorbar(xs, vals, yerr=sd, fmt="none", ecolor=INK_2, lw=1, capsize=0, label="± sd over scorer seeds")
    for i, s in enumerate(splits):
        br = sel["metrics"][s]["base_rate"]
        ax.plot([i - 0.42, i + 0.42], [br, br], color=MUTED, ls=(0, (4, 3)), lw=1.1, label="base rate (chance AP)" if i == 0 else None)
    diag = [r for r in res["runs"] if r.get("diagnostic")]
    for r in diag:
        ax.plot(np.arange(len(splits)) + 0.5 * width, [r["metrics"][s]["a"]["ap"] for s in splits], lw=0, marker="D", ms=5, color=INK,
                label=f"scorer, proj {r['proj_dim']}-d (diagnostic)")
    ax.set_xticks(range(len(splits)), [s.replace("val_", "val ").replace("_disjoint", "\nunseen wearer").replace("nlq", "NLQ").replace("qa", "QA") for s in splits], fontsize=8.5)
    ax.set_ylabel("average precision for $y_t$ (0–1)")
    ax.set_title("Average precision by split")
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=8.2)
    suptitle(fig, "Check 1 · does anything predict the moments people ask about?",
             f"frame-level labels y_t = 1 inside an answer window (±{cfg.get('training', {}).get('time_tolerance', 0.5)} s); "
             f"train frames {res['train_frames']}, positive rate {res['train_pos_rate']:.3f}")
    footer(fig, cfg_groups(cfg, {"Training": {"stage": "1 (scorer)", "loss": "weighted BCE", "pos_weight": round(res["pos_weight"], 3),
                                              "selected seed": res["selected"]["seed"], "dev AP": round(res["selected"]["dev_ap"], 4)}}), reserve_in=0.6)
    fig.subplots_adjust(right=0.82)
    save(fig, os.path.join(out_dir, "check1_signal.png"))

    if os.path.exists(scores):
        z = np.load(scores)
        if "example_t" in z:
            t = z["example_t"]
            fig, ax = plt.subplots(figsize=(14, 4.6))
            ax.fill_between(t, 0, z["example_y"], step="mid", color=SIGNAL_COLOR["window"], alpha=0.18, lw=0, label="answer window ($y_t$ = 1)")
            sb = z["example_sbar"]
            ax.plot(t, sb / max(sb.max(), 1e-9), color=SIGNAL_COLOR["sbar"], lw=1.4, label="$\\bar s_t$ / its max (arm A)")
            ax.plot(t, z["example_a"], color=SIGNAL_COLOR["a"], lw=0.8, alpha=0.7, label="scorer $a_t$")
            ax.plot(t, z["example_w"], color=SIGNAL_COLOR["w"], lw=1.6, label="EMA($a_t$) (arm B)")
            ax.set_ylim(-0.03, 1.05)
            ax.set_xlabel("stream time (s)")
            ax.set_ylabel("signal (0–1)")
            ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0))
            suptitle(fig, "Check 1 · write signals on one val stream", f"stream {str(z['example_id'])[:8]}, {len(t)} frames")
            footer(fig, cfg_groups(cfg), reserve_in=0.55)
            fig.subplots_adjust(right=0.82)
            save(fig, os.path.join(out_dir, "check1_stream.png"))


def plot_check2(run_dir: str, out_dir: str) -> None:
    path = os.path.join(run_dir, "replay", "check2.json")
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    cal = res["calibration"]
    tracks = [t for t in ("nlq", "qa") if t in res["tracks"]]
    arms = [a for a in ARM_ORDER if any(k.endswith("/" + a) for k in res["tracks"][tracks[0]])]
    Ns = sorted({int(k.split("/")[0][2:]) for k in res["tracks"][tracks[0]]})
    cal_txt = {a: ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in cal[a].items() if k != "note") for a in cal}
    extra = {"Write": {"replay": "closed-form nodes (mean g)", "reads": "not counted (p_acc = 0)"}, "Arms": cal_txt}

    fig, axes = plt.subplots(1, len(tracks) + 1, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1] * len(tracks) + [1.1]})
    fig.subplots_adjust(wspace=0.24)
    for ax, tr in zip(axes, tracks):
        r = res["tracks"][tr]
        for arm in arms:
            ax.plot(Ns, [r[f"N={n}/{arm}"]["best_iou"] for n in Ns], marker="o", ms=6, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ns, [str(n) for n in Ns])
        ax.set_xlabel("memory budget N (nodes)")
        ax.set_ylabel("best-node IoU at t_q (0–1)")
        ax.set_title(f"val {tr.upper()} track")
    axes[0].legend(loc="lower right", fontsize=8.5)
    ax = axes[-1]
    tr = tracks[0]
    r = res["tracks"][tr]
    others = [a for a in arms if a != "A"]
    w = 0.8 / max(1, len(others))
    for j, arm in enumerate(others):
        d = [r[f"N={n}/{arm}"]["vs_A"]["best_iou"] for n in Ns]
        xs = np.arange(len(Ns)) + (j - (len(others) - 1) / 2) * w
        ax.bar(xs, [x["mean"] for x in d], w * 0.9, color=ARM_COLOR[arm], edgecolor="#fcfcfb", lw=1, label=f"{arm} − A")
        for x0, x in zip(xs, d):
            ax.plot([x0, x0], x["ci95"], color=INK_2, lw=1)
    ax.axhline(0, color=AXIS, lw=1.2)
    ax.set_xticks(range(len(Ns)), [f"N={n}" for n in Ns])
    ax.set_ylabel("Δ best-node IoU vs surprise")
    ax.set_title(f"Headroom over surprise, val {tr.upper()} (95% CI)")
    ax.legend(loc="best", fontsize=8.5)
    suptitle(fig, "Check 2 · how much can the write signal change what memory keeps?",
             f"matched cut budget: {cal['A']['segments']:.1f} segments per train stream in every arm; C − A bounds what any signal can gain")
    footer(fig, cfg_groups({}, extra), width_chars=170, reserve_in=0.6)
    save(fig, os.path.join(out_dir, "check2_headroom.png"))

    n_main = 16 if 16 in Ns else Ns[0]
    buckets = [b for b in DELAY_ORDER if b in r[f"N={n_main}/A"]["by_delay"]]
    fig, axes = plt.subplots(1, len(tracks), figsize=(13, 5), squeeze=False)
    for ax, tr in zip(axes[0], tracks):
        r = res["tracks"][tr]
        for arm in arms:
            ax.plot(range(len(buckets)), [r[f"N={n_main}/{arm}"]["by_delay"].get(b, np.nan) for b in buckets], marker="o", ms=6,
                    color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.set_xticks(range(len(buckets)), [b[:-1] for b in buckets])
        ax.set_xlabel("delay t_q − t_end (s)")
        ax.set_ylabel("best-node IoU (0–1)")
        ax.set_title(f"val {tr.upper()} track, N = {n_main}")
    axes[0][-1].legend(loc="upper right", fontsize=8.5)
    suptitle(fig, "Check 2 · headroom by delay", "longer delays have been through more merges")
    footer(fig, cfg_groups({}, extra), width_chars=170, reserve_in=0.6)
    save(fig, os.path.join(out_dir, "check2_delay.png"))


def plot_q2(run_dir: str, out_dir: str) -> None:
    for path in sorted(glob.glob(os.path.join(run_dir, "q2_*", "q2_*_summary.json"))):
        s = json.load(open(path))
        ks = sorted(s["by_k"], key=int)
        if not ks:
            continue
        qa = s["track"] == "qa"
        fig, axes = plt.subplots(1, 4 if qa else 3, figsize=(17 if qa else 14, 5.6))
        fig.subplots_adjust(wspace=0.36)
        arms = ["single", "decomposed", "unmatched", "survival"]
        lab = {"single": "single retrieval (M)", "decomposed": "decomposed, M split (result)", "unmatched": "decomposed, k×M (diagnostic)",
               "survival": "kept in memory (ceiling)"}
        w = 0.8 / len(arms)
        for ax, metric, title in ((axes[0], "allhit", "All sub-question windows hit"), (axes[1], "subhit", "Sub-question windows hit")):
            for j, a in enumerate(arms):
                xs = np.arange(len(ks)) + (j - 1.5) * w
                vals = [s["by_k"][k][f"{a}_{metric}"] for k in ks]
                ax.bar(xs, vals, w * 0.92, color=Q2_COLOR[a], edgecolor="#fcfcfb", lw=1, label=lab[a], alpha=0.55 if a == "survival" else 1.0)
                for x0, v in zip(xs, vals):
                    ax.text(x0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.2, color=INK_2)
            ax.set_xticks(range(len(ks)), [f"k = {k}\n{s['by_k'][k]['items']} items" for k in ks])
            ax.set_ylim(0, 1.08)
            ax.set_ylabel("fraction (0–1)")
            ax.set_title(title, fontsize=10.5)
        handles = [Patch(color=Q2_COLOR[a], alpha=0.55 if a == "survival" else 1.0, label=lab[a]) for a in arms]
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, fig_legend_y(fig)), ncol=len(handles), fontsize=8.5)
        ax = axes[2]
        for i, k in enumerate(ks):
            c = s["by_k"][k]["evidence_check_curve"]
            ax.plot([p["coverage"] for p in c], [p["unsupported_rate"] for p in c], marker="o", ms=5, color=SERIES[i], label=f"k = {k}")
        ax.set_xlabel("coverage (fraction of sub-questions answered)")
        ax.set_ylabel("unsupported rate (0–1)")
        ax.set_title("Evidence check (abstain if GAR\nscore below threshold)", fontsize=10.5)
        ax.invert_xaxis()
        ax.legend(loc="upper right", fontsize=8.5)
        if qa:
            ax = axes[3]
            for j, a in enumerate(("single", "decomposed")):
                xs = np.arange(len(ks)) + (j - 0.5) * 0.36
                vals = [s["by_k"][k].get(f"{a}_contains") for k in ks]
                ax.bar(xs, vals, 0.34, color=Q2_COLOR[a], edgecolor="#fcfcfb", lw=1, label=lab[a])
                for x0, v in zip(xs, vals):
                    ax.text(x0, (v or 0) + 0.01, f"{v:.2f}" if v is not None else "–", ha="center", va="bottom", fontsize=7.5, color=INK_2)
            ax.set_xticks(range(len(ks)), [f"k = {k}" for k in ks])
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("sub-answers matched (0–1)")
            ax.set_title("Answers (contains-match)", fontsize=10.5)
        order = s["by_k"].get("2", {}).get("order_acc_decomposed")
        suptitle(fig, f"Q2 · decomposed vs single retrieval · arm {s['arm']} · val {s['track'].upper()} track",
                 f"same memory snapshot and evidence budget M = {s['M']}; sub-questions ≥ {s['min_gap_s']:.0f} s apart; ≤ {s['per_video']} items per video"
                 + (f"; order accuracy (k = 2, decomposed) {order:.2f} vs chance 0.50" if order is not None else ""))
        footer(fig, cfg_groups(s["config"], {"Data": {"track": s["track"], "split": "val"}}), reserve_in=0.75)
        save(fig, os.path.join(out_dir, f"q2_{s['track']}_{s['arm']}.png"), top_in=1.5)


# ----------------------------------------------------------------------------------------------- driver
def plot_traces(trace_dir: str, out_dir: str, animations: bool = True) -> None:
    for p in sorted(glob.glob(os.path.join(trace_dir, "*.json"))):
        tr = json.load(open(p))
        stem = os.path.splitext(os.path.basename(p))[0]
        plot_stream(tr, os.path.join(out_dir, f"{stem}_signals.png"))
        plot_nodes(tr, os.path.join(out_dir, f"{stem}_nodes.png"))
        if animations:
            animate_memory(tr, os.path.join(out_dir, f"{stem}_memory.gif"))
            animate_feature_map(tr, os.path.join(out_dir, f"{stem}_featuremap.gif"))
        print(f"[plot] {stem}")


def plot_run(run_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    steps = [("check1", lambda: plot_check1(run_dir, out_dir)), ("check2", lambda: plot_check2(run_dir, out_dir))]
    for log in sorted(glob.glob(os.path.join(run_dir, "*", "train_log.jsonl"))):
        steps.append((f"train {log}", lambda log=log: plot_train_log(log, out_dir)))
    steps.append(("stage W arms", lambda: plot_stage_w_arms(run_dir, out_dir)))
    for track in ("qa", "nlq"):
        def ev(track=track):
            evals = load_evals(run_dir, track)
            if evals:
                plot_eval_overview(evals, track, out_dir)
                plot_eval_vs_a(evals, track, out_dir)
                plot_eval_delay(evals, track, out_dir)
                plot_eval_slices(evals, track, out_dir)
        steps.append((f"eval {track}", ev))
    steps += [("budget", lambda: plot_budget(run_dir, out_dir)), ("q2", lambda: plot_q2(run_dir, out_dir))]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:  # one broken input must not stop the other figures
            print(f"[plot][WARN] {name}: {type(e).__name__}: {e}")
    print(f"[plot] run figures in {out_dir}: {len(glob.glob(os.path.join(out_dir, '*.png')))} png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="outputs/<RUN>: every figure the run's outputs allow, into <run_dir>/figures.")
    ap.add_argument("--traces", default=None, help="Directory of trace JSONs from train/eval --trace_dir.")
    ap.add_argument("--train_log", default=None)
    ap.add_argument("--config", default=None, help="YAML config, for the parameter footer of the training plot.")
    ap.add_argument("--summaries", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_animations", action="store_true")
    args = ap.parse_args()
    if args.run_dir:
        plot_run(args.run_dir, args.out or os.path.join(args.run_dir, "figures"))
    if not args.out:
        return
    os.makedirs(args.out, exist_ok=True)
    if args.traces:
        plot_traces(args.traces, args.out, not args.no_animations)
    if args.train_log:
        plot_train_log(args.train_log, args.out, args.config)
    if args.summaries:
        plot_summaries(args.summaries, args.out)


if __name__ == "__main__":
    main()
