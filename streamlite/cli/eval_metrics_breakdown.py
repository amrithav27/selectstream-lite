"""P2: Standard metrics and fine-grained breakdowns from predictions.jsonl.

Computes:
1. Lenient answer metrics: Exact match, Contains match, Normalized match, Token F1.
2. Temporal localization (NLQ-style): R@1 and R@5 at IoU >= 0.3 and IoU >= 0.5 from ranked evidence_spans vs window.
3. Breakdowns across:
   - Overall
   - Delay bucket
   - Participant disjoint vs seen
   - Linked vs unlinked
   - Template
   - Evidence location (open vs committed)
4. Paired bootstrap over videos (resampling stream_id) for every run compared to a reference run (e.g. W_A).

Usage:
    python -m streamlite.cli.eval_metrics_breakdown \
        --preds W_A=eval/W_A/predictions.jsonl W_B=eval/W_B/predictions.jsonl \
        --ref W_A \
        --out_dir outputs/breakdown_p2
"""
from __future__ import annotations
import argparse
import json
import os
import re
import string
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
# ---------------------------------------------------------------------------
# Answer Normalization & Metrics
# ---------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())
def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)
def compute_answer_metrics(pred: Optional[str], gold: Optional[str]) -> Dict[str, float]:
    p_norm = normalize_text(pred)
    g_norm = normalize_text(gold)
    p_raw = (pred or "").strip()
    g_raw = (gold or "").strip()
    exact = float(p_raw == g_raw and bool(g_raw))
    contains = float(bool(g_norm) and (g_norm in p_norm or (bool(p_norm) and p_norm in g_norm)))
    norm_match = float(p_norm == g_norm and bool(g_norm))
    token_f1 = compute_token_f1(pred or "", gold or "")
    return {
        "exact": exact,
        "contains": contains,
        "norm_match": norm_match,
        "token_f1": token_f1,
    }
# ---------------------------------------------------------------------------
# Temporal Span / Localization Metrics (NLQ style)
# ---------------------------------------------------------------------------
def span_iou(span: Sequence[float], window: Sequence[float]) -> float:
    s, e = float(span[0]), float(span[1])
    ws, we = float(window[0]), float(window[1])
    inter = max(0.0, min(e, we) - max(s, ws))
    union = max(e, we) - min(s, ws)
    return inter / union if union > 0.0 else 0.0
def compute_localization_metrics(evidence_spans: List[Sequence[float]], window: Sequence[float]) -> Dict[str, float]:
    if not evidence_spans or not window:
        return {
            "r1_iou03": 0.0,
            "r1_iou05": 0.0,
            "r5_iou03": 0.0,
            "r5_iou05": 0.0,
            "top1_iou": 0.0,
            "max_iou": 0.0,
        }
    ious = [span_iou(sp, window) for sp in evidence_spans]
    top1_iou = ious[0] if ious else 0.0
    top5_ious = ious[:5]
    max_iou = max(ious) if ious else 0.0
    return {
        "r1_iou03": float(top1_iou >= 0.3),
        "r1_iou05": float(top1_iou >= 0.5),
        "r5_iou03": float(any(x >= 0.3 for x in top5_ious)),
        "r5_iou05": float(any(x >= 0.5 for x in top5_ious)),
        "top1_iou": float(top1_iou),
        "max_iou": float(max_iou),
    }
# ---------------------------------------------------------------------------
# Row Parsing & Aggregation
# ---------------------------------------------------------------------------
METRIC_KEYS = [
    "exact", "contains", "norm_match", "token_f1",
    "r1_iou03", "r1_iou05", "r5_iou03", "r5_iou05", "top1_iou", "max_iou"
]
def parse_and_enrich_row(r: dict) -> dict:
    row = dict(r)
    ans_metrics = compute_answer_metrics(r.get("prediction"), r.get("answer"))
    loc_metrics = compute_localization_metrics(r.get("evidence_spans", []), r.get("window", []))
    row.update(ans_metrics)
    row.update(loc_metrics)
    return row
def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(parse_and_enrich_row(json.loads(line)))
    return rows
def aggregate_metrics(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS} | {"n": 0}
    out = {"n": len(rows)}
    for k in METRIC_KEYS:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out
# ---------------------------------------------------------------------------
# Paired Bootstrap by Video (Stream ID)
# ---------------------------------------------------------------------------
def paired_bootstrap_test(
    ref_rows: List[dict],
    eval_rows: List[dict],
    key: str,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Resample video stream_id clusters with replacement to compute paired diff CI and p-value."""
    ref_by_id = {r["query_id"]: r for r in ref_rows if "query_id" in r}
    eval_by_id = {r["query_id"]: r for r in eval_rows if "query_id" in r}
    common_qids = sorted(set(ref_by_id.keys()) & set(eval_by_id.keys()))
    if not common_qids:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "paired_n": 0}
    # Group paired differences by stream_id
    stream_diffs = defaultdict(list)
    for qid in common_qids:
        r_ref, r_eval = ref_by_id[qid], eval_by_id[qid]
        sid = r_ref.get("stream_id", "default")
        d = float(r_eval.get(key, 0.0)) - float(r_ref.get(key, 0.0))
        stream_diffs[sid].append(d)
    stream_ids = list(stream_diffs.keys())
    stream_means = [np.mean(stream_diffs[s]) for s in stream_ids]
    stream_counts = [len(stream_diffs[s]) for s in stream_ids]
    total_q = sum(stream_counts)
    # Base weighted mean difference
    obs_diff = sum(m * c for m, c in zip(stream_means, stream_counts)) / total_q
    rng = np.random.default_rng(seed)
    n_streams = len(stream_ids)
    boot_diffs = []
    for _ in range(n_resamples):
        sample_indices = rng.choice(n_streams, size=n_streams, replace=True)
        sample_sum = sum(stream_means[i] * stream_counts[i] for i in sample_indices)
        sample_cnt = sum(stream_counts[i] for i in sample_indices)
        boot_diffs.append(sample_sum / sample_cnt if sample_cnt > 0 else 0.0)
    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    p_le = np.mean(boot_diffs <= 0.0)
    p_ge = np.mean(boot_diffs >= 0.0)
    p_value = 2.0 * min(p_le, p_ge)
    return {
        "diff": float(obs_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, p_value)),
        "paired_n": len(common_qids),
    }
# ---------------------------------------------------------------------------
# Slicing / Breakdown Generators
# ---------------------------------------------------------------------------
BREAKDOWNS: List[Tuple[str, Callable[[dict], str]]] = [
    ("delay_bucket", lambda r: str(r.get("delay_bucket", "unknown"))),
    ("participant_disjoint", lambda r: "disjoint (unseen)" if r.get("participant_disjoint") else "seen"),
    ("linked", lambda r: "linked" if r.get("linked") else "unlinked"),
    ("evidence_location", lambda r: str(r.get("evidence_location", "unknown"))),
    ("template", lambda r: str(r.get("template", "unknown"))),
]
def compute_full_breakdown(rows: List[dict]) -> Dict[str, Any]:
    out = {"overall": aggregate_metrics(rows), "breakdowns": {}}
    for group_name, fn in BREAKDOWNS:
        grouped = defaultdict(list)
        for r in rows:
            grouped[fn(r)].append(r)
        out["breakdowns"][group_name] = {k: aggregate_metrics(g_rows) for k, g_rows in grouped.items()}
    return out
# ---------------------------------------------------------------------------
# Formatting Tables
# ---------------------------------------------------------------------------
def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    col_w = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    lines = [
        "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_w)) + " |",
        "| " + " | ".join("-" * w for w in col_w) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(r, col_w)) + " |")
    return "\n".join(lines)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="Prediction files as Name=path/to/predictions.jsonl or path/to/predictions.jsonl")
    ap.add_argument("--ref", default=None, help="Reference run name to compute paired bootstrap against (e.g. W_A).")
    ap.add_argument("--out_dir", default="outputs/metrics_breakdown_p2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # Parse inputs
    runs: Dict[str, List[dict]] = {}
    for item in args.preds:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = os.path.basename(os.path.dirname(item)) or os.path.splitext(os.path.basename(item))[0]
            path = item
        if not os.path.exists(path):
            raise FileNotFoundError(f"Predictions file not found: {path}")
        print(f"[P2] Loading {name} from {path} ...")
        runs[name] = load_jsonl(path)
    ref_name = args.ref
    if not ref_name or ref_name not in runs:
        matches = [k for k in runs if ref_name and (ref_name.lower() in k.lower() or k.lower() in ref_name.lower())]
        ref_name = matches[0] if matches else list(runs.keys())[0]
    print(f"[P2] Reference run for paired bootstrap: {ref_name}")
    results = {}
    table_overall = []
    headers_overall = [
        "Run", "N", "Exact", "Contains", "Token F1", "R@1 (0.3)", "R@1 (0.5)", "R@5 (0.3)", "R@5 (0.5)", "Top-1 IoU"
    ]
    for name, rows in runs.items():
        summary = compute_full_breakdown(rows)
        results[name] = summary
        ov = summary["overall"]
        table_overall.append([
            name,
            ov["n"],
            f"{ov['exact']:.3f}",
            f"{ov['contains']:.3f}",
            f"{ov['token_f1']:.3f}",
            f"{ov['r1_iou03']:.3f}",
            f"{ov['r1_iou05']:.3f}",
            f"{ov['r5_iou03']:.3f}",
            f"{ov['r5_iou05']:.3f}",
            f"{ov['top1_iou']:.3f}",
        ])
    md_report = ["# P2 Ablation: Standard Metrics & Breakdowns\n"]
    md_report.append("## Overall Metrics\n")
    md_report.append(format_markdown_table(headers_overall, table_overall))
    md_report.append("\n")
    # Bootstrap testing against ref
    if ref_name in runs and len(runs) > 1:
        md_report.append(f"## Paired Bootstrap vs Reference `{ref_name}` (resampling video streams, 95% CI)\n")
        boot_headers = ["Run", "Metric", f"Diff vs {ref_name}", "95% CI", "p-value"]
        boot_rows = []
        bootstrap_results = {}
        for name, rows in runs.items():
            if name == ref_name:
                continue
            bootstrap_results[name] = {}
            for k in ["contains", "token_f1", "r1_iou03", "r5_iou03", "top1_iou"]:
                b_res = paired_bootstrap_test(runs[ref_name], rows, key=k)
                bootstrap_results[name][k] = b_res
                boot_rows.append([
                    name,
                    k,
                    f"{b_res['diff']:+.4f}",
                    f"[{b_res['ci_low']:+.4f}, {b_res['ci_high']:+.4f}]",
                    f"{b_res['p_value']:.4f}",
                ])
        results["bootstrap_vs_ref"] = bootstrap_results
        md_report.append(format_markdown_table(boot_headers, boot_rows))
        md_report.append("\n")
    # Detailed breakdowns
    md_report.append("## Fine-Grained Breakdowns\n")
    for group_name, _ in BREAKDOWNS:
        md_report.append(f"### By `{group_name}`\n")
        cats = sorted(set(c for s in results.values() if "breakdowns" in s for c in s["breakdowns"].get(group_name, {}).keys()))
        brk_headers = ["Run", group_name, "N", "Contains", "Token F1", "R@1 (0.3)", "Top-1 IoU"]
        brk_rows = []
        for name in runs:
            for cat in cats:
                entry = results[name]["breakdowns"][group_name].get(cat, {})
                if entry.get("n", 0) > 0:
                    brk_rows.append([
                        name, cat, entry["n"],
                        f"{entry.get('contains', 0.0):.3f}",
                        f"{entry.get('token_f1', 0.0):.3f}",
                        f"{entry.get('r1_iou03', 0.0):.3f}",
                        f"{entry.get('top1_iou', 0.0):.3f}",
                    ])
        md_report.append(format_markdown_table(brk_headers, brk_rows))
        md_report.append("\n")
    report_text = "\n".join(md_report)
    print("\n" + report_text)
    # Save to disk
    out_json = os.path.join(args.out_dir, "breakdown_summary.json")
    out_md = os.path.join(args.out_dir, "breakdown_summary.md")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    with open(out_md, "w") as f:
        f.write(report_text)
    print(f"\n[P2] Saved summary JSON: {out_json}")
    print(f"[P2] Saved summary Markdown: {out_md}")
if __name__ == "__main__":
    main()
