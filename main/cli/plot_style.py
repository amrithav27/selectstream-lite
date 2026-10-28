"""Shared look for every SelectStream figure: one palette, recessive chrome, and a grouped parameter footer.

Colours are the validated default categorical palette (dataviz skill, references/palette.md), assigned to
entities in a fixed order and never by rank: the same arm has the same colour in every figure.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

ARM_COLOR = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2], "D": SERIES[3], "B_Rwriter": SERIES[4], "untrained": MUTED,
             "B0": SERIES[6], "E": SERIES[5]}
ARM_LABEL = {"A": "A · surprise (paper)", "B": "B · learned scorer", "C": "C · oracle labels", "D": "D · fixed length",
             "B_Rwriter": "B · Stage-R writer", "untrained": "untrained trainables", "B0": "B0 · scorer w/o surprise", "E": "E · surprise + scorer"}
ARM_TICK = {"B_Rwriter": "B\nR-writer", "untrained": "untr."}
ARM_ORDER = ["A", "B", "C", "D", "B_Rwriter", "untrained", "B0", "E"]
SIGNAL_COLOR = {"sbar": SERIES[0], "s": "#86b6ef", "w": SERIES[1], "a": SERIES[6], "theta": MUTED, "window": SERIES[2],
                "ask": SERIES[7], "cut": AXIS, "node": SERIES[0]}
Q2_COLOR = {"single": SERIES[0], "decomposed": SERIES[1], "unmatched": SERIES[2], "survival": MUTED}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": INK_2, "text.color": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 1.8, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9, "legend.handlelength": 1.6,
        "figure.dpi": 140, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


apply_style()


def arm_key(name: str) -> str:
    for k in sorted(ARM_LABEL, key=len, reverse=True):
        if name.endswith(k):
            return k
    return name


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        return f"{v:.4g}"
    return str(v)


def cfg_groups(cfg: dict, extra: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    """Every parameter that shapes a run, grouped for the footer."""
    m, v, d, t = cfg.get("model", {}), cfg.get("vismem", {}), cfg.get("data", {}), cfg.get("training", {})
    groups = {
        "Model": {"backbone": str(m.get("model_name_or_path", "?")).split("/")[-1] + " (frozen)", "dtype": m.get("torch_dtype"),
                  "Enc_q layers": v.get("query_builder", {}).get("num_layers"), "SegEnc layers": v.get("lvm_segment_encoder_layers"),
                  "GAR layers": v.get("method_gar_layers"), "trainables": v.get("trainable_dtype")},
        "Memory": {"N": v.get("lvm_num_slots"), "B": v.get("method_graph_budget"), "seeds": v.get("method_seed_topk"),
                   "M": v.get("method_evidence_tokens"), "τ_r": v.get("lvm_tau_r"), "τ_s": v.get("lvm_tau_s"),
                   "λ sim/sup/acc/rec": f"{v.get('lvm_merge_similarity_weight')}/{v.get('lvm_merge_surprise_weight')}/"
                                        f"{v.get('lvm_merge_access_weight')}/{v.get('lvm_merge_recency_weight')}"},
        "Write": {"signal": v.get("lvm_write_signal"), "L_min/L_max": f"{v.get('lvm_saw_l_min')}/{v.get('lvm_saw_l_max')}",
                  "B_s": v.get("lvm_saw_energy_budget"), "q/window": f"{v.get('lvm_saw_quantile')}/{v.get('lvm_saw_recent_window')}",
                  "ρ": v.get("lvm_saw_ema_decay"), "attn term": "off" if not v.get("lvm_saw_use_attn_proxy", True) else v.get("lvm_saw_lambda_attn"),
                  "scorer proj": v.get("lvm_scorer_proj_dim")},
        "Read": {"η": v.get("method_eta_surprise_boost"), "ξ_span/ξ_merge": f"{v.get('method_span_penalty_weight')}/{v.get('method_merge_penalty_weight')}",
                 "hops": v.get("method_num_hops"), "sim top-k": v.get("method_sim_topk")},
        "Training": {"lr": t.get("lr"), "epochs": t.get("epochs"), "grad accum": t.get("grad_accum"), "β/γ": f"{t.get('beta_ret')}/{t.get('gamma_spar')}",
                     "ε": t.get("time_tolerance"), "seed": t.get("seed")},
        "Data": {"fps": d.get("sample_fps"), "frame side": d.get("frame_max_side"), "t_q": d.get("query_time_mode"), "dev frac": d.get("dev_frac")},
    }
    if not m:
        groups = {g: {} for g in groups}  # no run config (e.g. the CPU replay): show only what the caller passes
    for g, kv in (extra or {}).items():
        groups.setdefault(g, {}).update(kv)
    return groups


def footer(fig, groups: Dict[str, Dict], width_chars: int = 150, reserve_in: float = 0.8) -> None:
    """Grouped parameter lines in a band below the axes (muted ink); `reserve_in` inches are left above the
    band for x labels and a legend placed under the axes."""
    lines: List[str] = []
    for g, kv in groups.items():
        items = [f"{k} {fmt(val)}" for k, val in kv.items() if val is not None and "None" not in fmt(val)]
        if not items:
            continue
        cur, first = f"{g:<10}", True
        for it in items:
            sep = "" if first else "  ·  "
            if not first and len(cur) + len(sep) + len(it) > width_chars:
                lines.append(cur)
                cur = " " * 10 + it
            else:
                cur += sep + it
            first = False
        lines.append(cur)
    h_in = fig.get_size_inches()[1]
    line_h = 0.15 / h_in
    band = line_h * (len(lines) + 1.6)
    fig.subplots_adjust(bottom=min(0.6, band + reserve_in / h_in))
    for i, ln in enumerate(lines):
        fig.text(0.01, 0.008 + line_h * (len(lines) - 1 - i), ln, fontsize=7.4, family="monospace", color=MUTED, ha="left", va="bottom")
    fig.text(0.01, 0.008 + line_h * len(lines) + 0.003, "Parameters", fontsize=7.8, color=INK_2, weight="semibold", ha="left", va="bottom")


def suptitle(fig, title: str, subtitle: Optional[str] = None) -> None:
    h = fig.get_size_inches()[1]
    fig.text(0.01, 1 - 0.10 / h, title, fontsize=13.5, weight="semibold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.01, 1 - 0.42 / h, subtitle, fontsize=9.5, color=INK_2, ha="left", va="top")


def fig_legend_y(fig) -> float:
    """y (figure fraction) for a figure-level legend row under the subtitle."""
    return 1 - 0.68 / fig.get_size_inches()[1]


def save(fig, path: str, top_in: float = 1.0) -> None:
    """top_in: inches kept above the axes for title, subtitle (and a figure legend, ~1.4)."""
    fig.subplots_adjust(top=1 - top_in / fig.get_size_inches()[1])
    fig.savefig(path)
    plt.close(fig)


def trailing_mean(vals: Sequence, k: int) -> np.ndarray:
    v = np.array([np.nan if a is None else a for a in vals], dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1) : i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if w.size else np.nan
    return out


def bootstrap_mean(rows: List[dict], key: str, group: str = "stream_id", n_boot: int = 1000, seed: int = 0):
    """Mean over queries, 95% CI from resampling videos (streams). None if no values."""
    by: Dict[str, List[float]] = {}
    for r in rows:
        if r.get(key) is not None:
            by.setdefault(r[group], []).append(float(r[key]))
    if not by:
        return None
    s = np.array([np.sum(v) for v in by.values()])
    c = np.array([len(v) for v in by.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)), int(c.sum())


def paired_diff(rows_a: List[dict], rows_b: List[dict], key: str, n_boot: int = 1000, seed: int = 0):
    """Mean of b - a over queries present in both (matched by query_id), 95% CI over videos."""
    a = {r["query_id"]: r for r in rows_a if r.get(key) is not None}
    diffs = [{"stream_id": r["stream_id"], "d": float(r[key]) - float(a[r["query_id"]][key])}
             for r in rows_b if r.get(key) is not None and r["query_id"] in a]
    return bootstrap_mean(diffs, "d", n_boot=n_boot, seed=seed)


def legend_below(ax, ncol: int = 4, handles=None, labels=None) -> None:
    kw = {"loc": "upper left", "bbox_to_anchor": (0, -0.16), "ncol": ncol, "fontsize": 9}
    if handles is not None:
        ax.legend(handles, labels, **kw)
    else:
        ax.legend(**kw)


def label_bar_ends(ax, bars, values: Iterable[float], fmtstr: str = "{:.2f}") -> None:
    for b, v in zip(bars, values):
        if v is None or np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmtstr.format(v), ha="center", va="bottom", fontsize=7.5, color=INK_2)
