#!/bin/bash
# scripts/colab_run_p3.sh
# Asynchronous, disconnect-resilient runner for P3:
# Evaluates Plain Qwen2.5-VL-7B (no memory) on Oracle and Uniform frames.
#
# Usage:
#   bash scripts/colab_run_p3.sh
# Or in background (appending to log with visual separator):
#   nohup bash scripts/colab_run_p3.sh >> /content/drive/MyDrive/selectstream_lite/logs/p3.log 2>&1 &

set -uo pipefail

# Directories
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive}"
VIDEO_DIR="${VIDEO_DIR:-$DRIVE_ROOT/ego4d_nlq/clips}"
WORK_DIR="${WORK_DIR:-$DRIVE_ROOT/selectstream_lite}"
OUT_ROOT="$WORK_DIR/eval"
LOG_DIR="$WORK_DIR/logs"
CACHE_DIR="${CACHE_DIR:-/content/feature_cache}"
CLEAN_DIR="${CLEAN_DIR:-$REPO_ROOT/dataset/clean}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/ego4d_colab.yaml}"
MAX_CLIPS="${MAX_CLIPS:-50}"

# Auto-detect fast local NVMe SSD clips if copied to /content/clips
if [ -d "/content/clips" ] && [ "$(ls -A /content/clips 2>/dev/null)" ]; then
    echo "[P3 Runner] ⚡ Detected fast local SSD clips at /content/clips! Using local disk."
    VIDEO_DIR="/content/clips"
fi

mkdir -p "$OUT_ROOT" "$LOG_DIR"

echo ""
echo "=========================================================="
echo "========== P3 RUN SESSION: $(date) =========="
echo "=========================================================="
echo "[P3 Runner] Video Directory: $VIDEO_DIR"
echo "[P3 Runner] Output Directory: $OUT_ROOT"
echo "[P3 Runner] Max Clips: $MAX_CLIPS"
echo "=========================================================="

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[P3 Runner][ERROR] Video directory not found: $VIDEO_DIR"
    exit 1
fi

done_() { [ -f "$LOG_DIR/done.p3_$1" ]; }
mark() { touch "$LOG_DIR/done.p3_$1"; echo "[P3 Runner] ===== P3 $1 done at $(date) ====="; }

# 1. Oracle Frames baseline (up to 8 frames from answer window)
if ! done_ oracle; then
    echo "[P3 Runner] Running Oracle Frames baseline (max_clips=$MAX_CLIPS, resume=enabled) ..."
    python3 -m main.cli.eval_ego4d \
        --config "$CONFIG" \
        --ckpt none \
        --track qa \
        --clean_dir "$CLEAN_DIR" \
        --video_dir "$VIDEO_DIR" \
        --feature_cache "$CACHE_DIR" \
        --frames_baseline oracle \
        --num_frames 8 \
        --max_clips "$MAX_CLIPS" \
        --resume \
        --out_dir "$OUT_ROOT/p3_oracle_frames" \
        || { echo "[P3 Runner][ERROR] Oracle frames baseline failed"; exit 1; }
    mark oracle
else
    echo "[P3 Runner] Oracle Frames baseline already completed, skipping."
fi

# 2. Uniform Frames baseline (8 frames evenly spaced from stream start to t_q)
if ! done_ uniform; then
    echo "[P3 Runner] Running Uniform Frames baseline (max_clips=$MAX_CLIPS, resume=enabled) ..."
    python3 -m main.cli.eval_ego4d \
        --config "$CONFIG" \
        --ckpt none \
        --track qa \
        --clean_dir "$CLEAN_DIR" \
        --video_dir "$VIDEO_DIR" \
        --feature_cache "$CACHE_DIR" \
        --frames_baseline uniform \
        --num_frames 8 \
        --max_clips "$MAX_CLIPS" \
        --resume \
        --out_dir "$OUT_ROOT/p3_uniform_frames" \
        || { echo "[P3 Runner][ERROR] Uniform frames baseline failed"; exit 1; }
    mark uniform
else
    echo "[P3 Runner] Uniform Frames baseline already completed, skipping."
fi

echo "=========================================================="
echo "[P3 Runner] All P3 baselines finished successfully at $(date)!"
echo "=========================================================="
