"""Shared pieces of the Ego4D NLQ / QaEgo4D CLIs (train_ego4d, eval_ego4d, eval_q2, calibrate_write)."""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from main.cli.common import build_vismem_config, load_yaml
from main.model.model import VisMemModel
from main.utils.misc import to_torch_dtype
from main.utils.qwen_vl import load_qwen25vl

ARM_SIGNAL = {"A": "surprise", "B": "learned", "C": "oracle", "D": "fixed"}
TRAINABLE_GROUPS = ("reader", "writer", "scorer")

PROMPT_TEMPLATE = (
    "You are watching a first-person video stream. Answer the question about what happened earlier "
    "in the video in a few words.\nQuestion: {question}"
)


def format_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question.strip())


CALIBRATED_KEYS = ("lvm_saw_energy_budget", "lvm_saw_quantile", "lvm_tau_s", "lvm_tau_r", "lvm_fixed_segment_len")


def apply_arm(cfg_dict: Dict[str, Any], arm: str, calibration: Optional[str] = None) -> Dict[str, Any]:
    """Arm A-D sets the write signal and its thresholds: configs `arms:` block, then overridden by
    the calibrate_write output when `calibration` points to an existing file."""
    if arm not in ARM_SIGNAL:
        raise ValueError(f"arm must be one of {sorted(ARM_SIGNAL)}, got {arm!r}")
    v = cfg_dict["vismem"]
    v["lvm_write_signal"] = ARM_SIGNAL[arm]
    values = dict(cfg_dict.get("arms", {}).get(arm) or {})
    # Arm A takes only its calibrated tau_s / tau_r (its energy budget is the paper's); B/C/D take all keys.
    keys = ("lvm_tau_s", "lvm_tau_r") if arm == "A" else CALIBRATED_KEYS
    if calibration and os.path.exists(calibration):
        with open(calibration) as f:
            cal = json.load(f).get(arm, {})
        values.update({k: cal[k] for k in keys if cal.get(k) is not None})
        print(f"[arm] {arm}: calibrated values from {calibration}: {values}")
    else:
        print(f"[arm][WARN] {arm}: no calibration file ({calibration}); using config values {values}")
    for key, value in values.items():
        v[key] = value
    cfg_dict["arm_values"] = {"arm": arm, **values}
    return cfg_dict


def load_model(cfg_dict: Dict[str, Any], ckpt: Optional[str] = None) -> VisMemModel:
    mcfg = cfg_dict["model"]
    max_memory = mcfg.get("max_memory", None)
    if max_memory is not None:
        max_memory = {int(k) if str(k).isdigit() else k: v for k, v in max_memory.items()}
    base_model, tokenizer, processor = load_qwen25vl(
        mcfg["model_name_or_path"],
        torch_dtype=to_torch_dtype(mcfg.get("torch_dtype", "float16")),
        device_map=mcfg.get("device_map", "auto"),
        trust_remote_code=bool(mcfg.get("trust_remote_code", True)),
        max_memory=max_memory,
    )
    for p in base_model.parameters():
        p.requires_grad = False
    model = VisMemModel(base_model, tokenizer, processor, build_vismem_config(cfg_dict))
    if ckpt is not None:
        load_trainable_state(model, ckpt)
    return model


def module_groups(model: VisMemModel) -> Dict[str, List[torch.nn.Parameter]]:
    """reader = Enc_q + GAR + W_e; writer = SegEnc + gate + f_write + slot init; scorer = Q1 scorer."""
    scorer = model.lvm_short_former.importance_scorer
    scorer_ids = {id(p) for p in scorer.parameters()} if scorer is not None else set()
    writer = [p for p in model.lvm_short_former.parameters() if id(p) not in scorer_ids]
    reader = list(model.query_builder.parameters()) + list(model.graph_reasoner.parameters())
    return {"reader": reader, "writer": writer, "scorer": list(scorer.parameters()) if scorer is not None else []}


def set_trainable(model: VisMemModel, groups: Iterable[str]) -> List[torch.nn.Parameter]:
    groups = set(groups)
    unknown = groups - set(TRAINABLE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown module groups: {sorted(unknown)}")
    for p in model.parameters():
        p.requires_grad = False
    out = []
    for name, params in module_groups(model).items():
        for p in params:
            p.requires_grad = name in groups
            if name in groups:
                out.append(p)
    return out


def trainable_state_dict(model: VisMemModel) -> Dict[str, torch.Tensor]:
    # Never save the frozen 7B backbone.
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("base_model.")}


def load_trainable_state(model: VisMemModel, ckpt: str) -> None:
    path = ckpt
    if not ckpt.endswith(".pt"):
        # weights.pt (no optimizer state) is what gets archived; trainable.pt is the resumable one.
        path = os.path.join(ckpt, "weights.pt")
        if not os.path.exists(path):
            path = os.path.join(ckpt, "trainable.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    payload = state.get("model", state)
    # Memory-budget ablation: evaluating at another N than trained. slot_init ([N, D], the learned
    # initial vector of each free slot) is the only N-shaped tensor; row i takes trained row i mod N_ckpt.
    key = "lvm_short_former.slot_init"
    if key in payload and key in model.state_dict() and payload[key].shape != model.state_dict()[key].shape:
        n_new, n_old = model.state_dict()[key].shape[0], payload[key].shape[0]
        payload = dict(payload)
        payload[key] = payload[key][torch.arange(n_new) % n_old]
        print(f"[ckpt] {key}: tiled {n_old} -> {n_new} slots (eval-time N override)")
    missing, unexpected = model.load_state_dict(payload, strict=False)
    missing = [k for k in missing if not k.startswith("base_model.")]
    print(f"[ckpt] loaded {path}: {len(payload)} tensors, missing(non-backbone)={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"[ckpt] missing keys (first 10): {missing[:10]}")


def trace_extras(model: VisMemModel, state, img, g, k: int, every: int = 4) -> Dict[str, Any]:
    """Per-frame trace additions for the figures: committed node spans (memory animation) and, when the
    write path ran from the cache, the visual-token norm map of every `every`-th frame (feature-map GIF)."""
    from main.data.frame_cache import as_image

    view = model.lvm_short_former.state_view(state[0])
    out: Dict[str, Any] = {"node_spans": torch.stack([view["active_time_starts"], view["active_time_ends"]], -1).detach().float().cpu().tolist()}
    if g is not None and k % every == 0:
        with torch.no_grad():
            H, grid = model.encode_frame(as_image(img))
        out["token_norms"] = H[0].detach().norm(dim=-1).float().cpu().tolist()
        out["grid"] = grid
    return out


def prepare_scorer(model: VisMemModel, arm: str, path: Optional[str]) -> None:
    """Load the Stage-1 scorer (main/cli/train_scorer.py) over whatever a checkpoint held. Call after
    every other weight load. Arm B without it would write with an untrained network, so that fails."""
    if not path:
        if arm == "B":
            raise ValueError("Arm B needs --scorer (the Stage-1 scorer); the checkpoint's scorer is untrained.")
        return
    scorer = model.lvm_short_former.importance_scorer
    if scorer is None:
        raise ValueError("--scorer given but the model was built without an importance scorer.")
    state = torch.load(path, map_location="cpu", weights_only=False)
    scorer.load_state_dict(state["scorer"])
    print(f"[scorer] loaded {path} (seed {state.get('seed')}, dev AP {state.get('dev_ap')})")


def atomic_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def frame_labels(times: Sequence[float], windows: Sequence[Tuple[float, float]], tol: float) -> np.ndarray:
    """y_t = 1[t inside any answer window, widened by tol] (Eq. 55's epsilon)."""
    t = np.asarray(times, dtype=np.float64)
    y = np.zeros_like(t)
    for s, e in windows:
        y[(t >= s - tol) & (t <= e + tol)] = 1.0
    return y


def scorer_pos_weight(items, tol: float) -> float:
    """#neg / #pos frames over the training streams, from the annotations alone (no video)."""
    pos = neg = 0.0
    for item in items:
        y = frame_labels(item.frame_times, [(q.window_start, q.window_end) for q in item.queries], tol)
        pos += float(y.sum())
        neg += float(len(y) - y.sum())
    return neg / max(pos, 1.0)


DELAY_EDGES = (0, 16, 64, 256, 1024)


def delay_bucket(q) -> str:
    """t_q - t_end in seconds; [0,16) is mostly still in the open segment (L_max = 64 frames)."""
    d = q.ask_time - q.window_end
    for lo, hi in zip(DELAY_EDGES, DELAY_EDGES[1:]):
        if lo <= d < hi:
            return f"[{lo},{hi})s"
    return f"[{DELAY_EDGES[-1]},inf)s" if d >= DELAY_EDGES[-1] else "<0s"


def train_participants(clean_dir: str) -> set:
    with open(os.path.join(clean_dir, "clips.json")) as f:
        return {c["participant_id"] for c in json.load(f) if c["split"] == "train" and c["participant_id"] is not None}


def log_device(prefix: str = "[env]") -> None:
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"{prefix} GPUs: {names}")
