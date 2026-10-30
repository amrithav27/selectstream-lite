#!/bin/bash
# scripts/colab_run_p4.sh
# Asynchronous, disconnect-resilient runner for P4:
# Stage W writer retraining across two additional random seeds (43, 44) for arms A and B,
# followed by evaluation on QA and NLQ.
#
# Resilience Features:
# - Self-contained background execution (safe against internet drops).
# - Automatic resume from latest/ checkpoint if a session disconnects mid-training.
# - Step completion markers (done.step_name) to skip finished stages on restart.
# - Video directory ($VIDEO_DIR) is treated as STRICTLY READ-ONLY.
# - Checkpoints, outputs, and logs are saved to Google Drive ($WORK_DIR).
#
# Usage:
#   nohup bash scripts/colab_run_p4.sh > /content/drive/MyDrive/selectstream_lite/logs/p4.log 2>&1 &

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Path definitions
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"
VIDEO_DIR="${VIDEO_DIR:-$DRIVE_ROOT/ego4d_nlq/clips}"
# Auto-detect fast local NVMe SSD clips if copied to /content/clips
if [ -d "/content/clips" ] && [ "$(ls -A /content/clips 2>/dev/null)" ]; then
    echo "[P4 Runner] ⚡ Detected fast local SSD clips at /content/clips! Using local disk."
    VIDEO_DIR="/content/clips"
fi
WORK_DIR="${WORK_DIR:-$DRIVE_ROOT/selectstream_lite}"
CKPT_ROOT="$WORK_DIR/checkpoints"
OUT_ROOT="$WORK_DIR/eval"
LOG_DIR="$WORK_DIR/logs"

# Hub checkpoint downloads
CK_DIR="${CK_DIR:-/content/ck}"
R_A_CKPT="${R_A_CKPT:-$CK_DIR/checkpoints/full_v3/R_A}"
SCORER="${SCORER:-$CK_DIR/runs/full_v3/scorer/scorer.pt}"
CAL="${CAL:-$CK_DIR/runs/full_v3/calibration.json}"
CACHE_DIR="${CACHE_DIR:-/content/feature_cache}"
CLEAN_DIR="${CLEAN_DIR:-$REPO_ROOT/dataset/clean}"

# Training budget
MAX_STEPS="${MAX_STEPS:-50}"
EVAL_CLIPS="${EVAL_CLIPS:-50}"   # matches the 50-stream evaluation budget
EVAL_TRACKS="${EVAL_TRACKS:-qa nlq}"

mkdir -p "$CKPT_ROOT" "$OUT_ROOT" "$LOG_DIR"

echo "=========================================================="
echo "[P4 Runner] Multi-seed Training (Seeds 43 & 44, Arms A & B)"
echo "[P4 Runner] Start time: $(date)"
echo "[P4 Runner] Video directory (READ-ONLY): $VIDEO_DIR"
echo "[P4 Runner] Working / checkpoint directory: $WORK_DIR"
echo "[P4 Runner] Initializing Stage W from: $R_A_CKPT"
echo "=========================================================="

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[P4 Runner][ERROR] Video directory not found: $VIDEO_DIR"
    exit 1
fi

done_() { [ -f "$LOG_DIR/done.p4_$1" ]; }
mark() { touch "$LOG_DIR/done.p4_$1"; echo "[P4 Runner] ===== P4 $1 completed at $(date) ====="; }

# Helper to train Stage W for a given arm and seed
train_stage_w() {
    local arm="$1"
    local seed="$2"
    local tag="W_${arm}_seed${seed}"
    local out_dir="$CKPT_ROOT/$tag"
    local config="$REPO_ROOT/configs/ego4d_colab_seed${seed}.yaml"

    if done_ "train_${tag}" || [ -f "$out_dir/final/weights.pt" ] || [ -f "$out_dir/weights.pt" ]; then
        echo "[P4 Runner] Stage W for $tag is already trained. Skipping training."
        mark "train_${tag}"
        return 0
    fi

    # Resume from latest/ if it exists, otherwise start fresh from R_A
    local init="$R_A_CKPT"
    if [ -f "$out_dir/latest/trainable.pt" ] || [ -f "$out_dir/latest/weights.pt" ]; then
        echo "[P4 Runner] Found existing latest/ checkpoint in $out_dir. Resuming training ..."
        init="$out_dir/latest"
    fi

    echo "[P4 Runner] >>> Starting Stage W training for $tag (arm=$arm, seed=$seed, init=$init) ..."
    python3 -m streamlite.cli.train_ego4d \
        --config "$config" \
        --stage W \
        --arm "$arm" \
        --seed "$seed" \
        --init_from "$init" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$out_dir" \
        --clean_dir "$CLEAN_DIR" \
        --video_dir "$VIDEO_DIR" \
        --feature_cache "$CACHE_DIR" \
        --scorer "$SCORER" \
        --calibration "$CAL" \
        --save_every_clips 10 \
        || { echo "[P4 Runner][ERROR] Training failed for $tag"; return 1; }

    mark "train_${tag}"
}

# Helper to evaluate a trained checkpoint
eval_checkpoint() {
    local arm="$1"
    local seed="$2"
    local track="$3"
    local tag="W_${arm}_seed${seed}"
    local ckpt="$CKPT_ROOT/$tag"
    local eval_tag="${tag}_${track}"
    local eval_out="$OUT_ROOT/$eval_tag"
    # IMPORTANT: Evaluates with original config (seed 42) so ask times are consistent across all runs
    local eval_config="$REPO_ROOT/configs/ego4d_colab.yaml"

    if done_ "eval_${eval_tag}"; then
        echo "[P4 Runner] Evaluation for $eval_tag already done. Skipping."
        return 0
    fi

    # Locate checkpoint: prefer final/weights.pt, then weights.pt, then latest/trainable.pt
    if [ -f "$ckpt/final/weights.pt" ]; then
        ckpt="$ckpt/final/weights.pt"
    elif [ -f "$ckpt/weights.pt" ]; then
        ckpt="$ckpt/weights.pt"
    elif [ -f "$ckpt/latest/trainable.pt" ]; then
        ckpt="$ckpt/latest/trainable.pt"
    fi

    echo "[P4 Runner] >>> Evaluating $eval_tag (track=$track, ckpt=$ckpt, max_clips=$EVAL_CLIPS) ..."
    python3 -m streamlite.cli.eval_ego4d \
        --config "$eval_config" \
        --arm "$arm" \
        --ckpt "$ckpt" \
        --track "$track" \
        --clean_dir "$CLEAN_DIR" \
        --video_dir "$VIDEO_DIR" \
        --feature_cache "$CACHE_DIR" \
        --scorer "$SCORER" \
        --calibration "$CAL" \
        --max_clips "$EVAL_CLIPS" \
        --resume \
        --out_dir "$eval_out" \
        || { echo "[P4 Runner][ERROR] Eval failed for $eval_tag"; return 1; }

    mark "eval_${eval_tag}"
}

# Execute Training & Evals: default to Seed 43 (Arms A & B)
P4_SEEDS="${P4_SEEDS:-43}"
for SEED in $P4_SEEDS; do
    for ARM in A B; do
        train_stage_w "$ARM" "$SEED"
        for TRACK in $EVAL_TRACKS; do
            eval_checkpoint "$ARM" "$SEED" "$TRACK"
        done
    done
done

echo "=========================================================="
echo "[P4 Runner] All P4 training and evaluations completed at $(date)!"
echo "=========================================================="

