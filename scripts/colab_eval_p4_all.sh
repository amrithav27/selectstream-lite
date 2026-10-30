#!/bin/bash
# scripts/colab_eval_p4_all.sh
# Batch evaluation runner for P4 checkpoints (W_A_seed43, W_B_seed43, etc.)
# Run on Colab after training completes.
#
# Features:
# - Evaluates on original config (ego4d_colab.yaml) as required by ablations_for_team.md
# - Resumes automatically if interrupted
# - Logs progress cleanly
#
# Usage:
#   bash scripts/colab_eval_p4_all.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"
WORK_DIR="${WORK_DIR:-$DRIVE_ROOT/selectstream_lite}"
CKPT_ROOT="$WORK_DIR/checkpoints"
OUT_ROOT="$WORK_DIR/eval"
LOG_DIR="$WORK_DIR/logs"

# Use local NVMe clips if available, else Drive clips
VIDEO_DIR="${VIDEO_DIR:-$DRIVE_ROOT/ego4d_nlq/clips}"
if [ -d "/content/clips" ] && [ "$(ls -A /content/clips 2>/dev/null)" ]; then
    VIDEO_DIR="/content/clips"
fi

CK_DIR="${CK_DIR:-/content/ck}"
SCORER="${SCORER:-$CK_DIR/runs/full_v3/scorer/scorer.pt}"
CAL="${CAL:-$CK_DIR/runs/full_v3/calibration.json}"
CACHE_DIR="${CACHE_DIR:-/content/feature_cache}"
CLEAN_DIR="${CLEAN_DIR:-$REPO_ROOT/dataset/clean}"

EVAL_CLIPS="${EVAL_CLIPS:-50}"
EVAL_CONFIG="$REPO_ROOT/configs/ego4d_colab.yaml"

# Target checkpoints to evaluate: default to Seed 43 (Arm A and Arm B)
P4_EVAL_TARGETS="${P4_EVAL_TARGETS:-W_A_seed43 W_B_seed43}"
EVAL_TRACKS="${EVAL_TRACKS:-qa nlq}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo "=========================================================="
echo "[P4 Batch Eval] Starting evaluation of P4 checkpoints"
echo "[P4 Batch Eval] Targets: $P4_EVAL_TARGETS"
echo "[P4 Batch Eval] Tracks:  $EVAL_TRACKS"
echo "[P4 Batch Eval] Clips:   $EVAL_CLIPS streams"
echo "[P4 Batch Eval] Video dir: $VIDEO_DIR"
echo "=========================================================="

for TAG in $P4_EVAL_TARGETS; do
    # Extract arm from tag (e.g., W_A_seed43 -> arm A)
    ARM="A"
    if [[ "$TAG" == *"_B_"* ]]; then
        ARM="B"
    fi
    CKPT="$CKPT_ROOT/$TAG"

    if [ ! -f "$CKPT/weights.pt" ] && [ ! -f "$CKPT/latest/weights.pt" ]; then
        echo "[P4 Batch Eval][WARNING] Checkpoint $CKPT not found! Skipping $TAG."
        continue
    fi

    for TRACK in $EVAL_TRACKS; do
        EVAL_TAG="${TAG}_${TRACK}"
        EVAL_OUT="$OUT_ROOT/$EVAL_TAG"
        DONE_MARKER="$LOG_DIR/done.eval_${EVAL_TAG}"

        if [ -f "$DONE_MARKER" ]; then
            echo "[P4 Batch Eval] $EVAL_TAG already marked as done. Skipping."
            continue
        fi

        echo ""
        echo "[P4 Batch Eval] >>> Evaluating $EVAL_TAG (arm=$ARM, track=$TRACK) at $(date) ..."
        python3 -m streamlite.cli.eval_ego4d \
            --config "$EVAL_CONFIG" \
            --arm "$ARM" \
            --ckpt "$CKPT" \
            --track "$TRACK" \
            --clean_dir "$CLEAN_DIR" \
            --video_dir "$VIDEO_DIR" \
            --feature_cache "$CACHE_DIR" \
            --scorer "$SCORER" \
            --calibration "$CAL" \
            --max_clips "$EVAL_CLIPS" \
            --resume \
            --out_dir "$EVAL_OUT"

        if [ $? -eq 0 ]; then
            touch "$DONE_MARKER"
            echo "[P4 Batch Eval] ✅ Finished $EVAL_TAG successfully at $(date)."
        else
            echo "[P4 Batch Eval][ERROR] Eval encountered an issue on $EVAL_TAG. Can be resumed."
        fi
    done
done

echo ""
echo "=========================================================="
echo "[P4 Batch Eval] All evaluations finished at $(date)!"
echo "=========================================================="

