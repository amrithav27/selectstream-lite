"""Resume checkpoints for the training loop: model, optimizer, position in the data and RNG state.

Layout under <output_dir>/ckpt/:
    step00000012/state.pt     one directory per checkpoint (written to *.tmp first, then renamed)
    latest                    text file with the name of the newest complete checkpoint

A checkpoint is only written right after an optimizer step, so no gradient is half accumulated, and it stores
`epoch` and `next_row`: the next row to train on. The data order is fixed, so resuming at that row never repeats
or skips a row.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import shutil
from typing import Any, Dict, Optional

import torch

_STEP_DIR = re.compile(r"^step(\d{8})$")


def dataset_fingerprint(ds) -> str:
    h = hashlib.sha1()
    for i in range(len(ds)):
        h.update(str(ds[i].id).encode("utf-8") + b"\0")
    return f"{len(ds)}:{h.hexdigest()[:16]}"


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np

        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _complete_dirs(root: str):
    if not os.path.isdir(root):
        return []
    names = [d for d in os.listdir(root) if _STEP_DIR.match(d) and os.path.exists(os.path.join(root, d, "state.pt"))]
    return sorted(names)


def find_latest(output_dir: str) -> Optional[str]:
    root = os.path.join(output_dir, "ckpt")
    pointer = os.path.join(root, "latest")
    if os.path.exists(pointer):
        with open(pointer) as f:
            name = f.read().strip()
        if os.path.exists(os.path.join(root, name, "state.pt")):
            return os.path.join(root, name)
    names = _complete_dirs(root)
    return os.path.join(root, names[-1]) if names else None


def save_resume_checkpoint(output_dir: str, model, opt, *, global_step: int, epoch: int, next_row: int,
                           fingerprint: str, grad_accum: int, log_lines: int, keep_last: int = 2) -> str:
    root = os.path.join(output_dir, "ckpt")
    os.makedirs(root, exist_ok=True)
    name = f"step{global_step:08d}"
    tmp, final = os.path.join(root, name + ".tmp"), os.path.join(root, name)
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    # The frozen backbone is reloaded from `model_name_or_path`; keep only the SelectStream modules.
    vismem_state = {k: v for k, v in model.state_dict().items() if not k.startswith("base_model.")}
    torch.save(
        {
            "vismem_state": vismem_state,
            "optimizer": opt.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "next_row": next_row,
            "fingerprint": fingerprint,
            "grad_accum": grad_accum,
            "log_lines": log_lines,
            "rng": _rng_state(),
        },
        os.path.join(tmp, "state.pt"),
    )
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    with open(os.path.join(root, "latest.tmp"), "w") as f:
        f.write(name)
    os.replace(os.path.join(root, "latest.tmp"), os.path.join(root, "latest"))
    if keep_last > 0:
        for old in _complete_dirs(root)[:-keep_last]:
            shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    return final


def load_resume_checkpoint(path: str, model, opt, *, fingerprint: str, grad_accum: int) -> Dict[str, Any]:
    """Restore model, optimizer and RNG from `path` (a step directory); returns the saved position."""
    file = os.path.join(path, "state.pt") if os.path.isdir(path) else path
    if not os.path.exists(file):
        raise FileNotFoundError(f"No resume checkpoint at {file}")
    state = torch.load(file, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise RuntimeError(
            f"Checkpoint was made on different training data (saved {state['fingerprint']}, now {fingerprint}); "
            "resuming would repeat or skip rows."
        )
    if state["grad_accum"] != grad_accum:
        raise RuntimeError(f"Checkpoint used grad_accum={state['grad_accum']}, this run uses {grad_accum}.")
    incompatible = model.load_state_dict(state["vismem_state"], strict=False)
    opt.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"])
    return {
        "global_step": int(state["global_step"]),
        "epoch": int(state["epoch"]),
        "next_row": int(state["next_row"]),
        "log_lines": int(state["log_lines"]),
        "missing": len(incompatible.missing_keys),
        "unexpected": len(incompatible.unexpected_keys),
    }


def truncate_log(path: str, keep_lines: int) -> None:
    """Drop log lines written after the checkpoint (those rows are trained again on resume)."""
    with open(path) as f:
        lines = f.readlines()
    with open(path, "w") as f:
        f.writelines(lines[:keep_lines])


# ---- Google Drive mirror -------------------------------------------------------------------------
# On Colab the Drive is a mounted folder (/content/drive/MyDrive/...), so a mirror is a plain copy. The local
# disk is lost when the runtime resets; the Drive copy is what a new session resumes from.

def _step_of(path: str) -> int:
    m = _STEP_DIR.match(os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def _copy_atomic(src: str, dst: str) -> None:
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    os.replace(tmp, dst)


def sync_to_drive(step_dir: str, output_dir: str, drive_dir: str, keep_last: int = 2) -> str:
    """Copy a finished step directory (and the loss log) to `drive_dir`, then move the `latest` pointer."""
    root = os.path.join(drive_dir, "ckpt")
    os.makedirs(root, exist_ok=True)
    name = os.path.basename(step_dir.rstrip("/"))
    _copy_atomic(step_dir, os.path.join(root, name))
    with open(os.path.join(root, "latest.tmp"), "w") as f:
        f.write(name)
    os.replace(os.path.join(root, "latest.tmp"), os.path.join(root, "latest"))
    log = os.path.join(output_dir, "loss_log.csv")
    if os.path.exists(log):
        shutil.copyfile(log, os.path.join(drive_dir, "loss_log.csv.tmp"))
        os.replace(os.path.join(drive_dir, "loss_log.csv.tmp"), os.path.join(drive_dir, "loss_log.csv"))
    if keep_last > 0:
        for old in _complete_dirs(root)[:-keep_last]:
            shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    return os.path.join(root, name)


def restore_from_drive(output_dir: str, drive_dir: str) -> Optional[str]:
    """Pull the Drive checkpoint into <output_dir>/ckpt if it is newer than the local one; return its local path."""
    drive_latest = find_latest(drive_dir)
    if drive_latest is None:
        return None
    local_latest = find_latest(output_dir)
    if local_latest is not None and _step_of(local_latest) >= _step_of(drive_latest):
        return local_latest
    root = os.path.join(output_dir, "ckpt")
    os.makedirs(root, exist_ok=True)
    name = os.path.basename(drive_latest)
    _copy_atomic(drive_latest, os.path.join(root, name))
    with open(os.path.join(root, "latest"), "w") as f:
        f.write(name)
    drive_log, local_log = os.path.join(drive_dir, "loss_log.csv"), os.path.join(output_dir, "loss_log.csv")
    if os.path.exists(drive_log):
        shutil.copyfile(drive_log, local_log)
    return os.path.join(root, name)


def copy_dir_to_drive(src_dir: str, drive_dir: str) -> str:
    """Mirror a plain folder (e.g. the end-of-epoch main.pt) to <drive_dir>/<folder name>."""
    dst = os.path.join(drive_dir, os.path.basename(src_dir.rstrip("/")))
    os.makedirs(drive_dir, exist_ok=True)
    _copy_atomic(src_dir, dst)
    return dst
