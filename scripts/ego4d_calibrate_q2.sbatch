#!/bin/bash
# After Stage R: (1) calibrate arm B/C/D thresholds (prints the `arms:` YAML block to paste into
# configs/ego4d_qwen25vl7b.yaml), and/or (2) Q2 decomposed-vs-single retrieval on NLQ val.
#   TASK=calibrate CKPT=$OUT_ROOT/R_A/epoch2       sbatch -w gnode061 scripts/ego4d_calibrate_q2.sbatch
#   TASK=q2 ARM=A CKPT=$OUT_ROOT/W_A/epoch0 K=2      sbatch -w gnode061 scripts/ego4d_calibrate_q2.sbatch
#SBATCH --job-name=ss_calq2
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
source scripts/ego4d_env.sh || exit 1
TASK="${TASK:?set TASK=calibrate or q2}"
CKPT="${CKPT:-}"
if [ "${SKIP_STAGE:-0}" = 1 ]; then echo "[calq2] SKIP_STAGE=1: using clips already staged"; else stage_clips || exit 1; fi
mkdir -p "$REPO_ROOT/outputs"
if [ "$TASK" = "calibrate" ]; then
    # No CKPT = pre-Stage-R probe: only arm A tau_s / tau_r are meaningful (frozen features only).
    "$PY" -m streamlite.cli.calibrate_write ${CKPT:+--ckpt "$CKPT"} --video_dir "$VIDEO_DIR" --num_clips "${NUM_CLIPS:-50}" \
        --out "$REPO_ROOT/outputs/calibration.json"
else
    ARM="${ARM:-A}"
    K="${K:-2}"
    [ -n "$CKPT" ] || { echo "[calq2] TASK=q2 needs CKPT"; exit 1; }
    "$PY" -m streamlite.cli.eval_q2 --arm "$ARM" --ckpt "$CKPT" --k "$K" ${MAX_CLIPS:+--max_clips "$MAX_CLIPS"} --video_dir "$VIDEO_DIR" \
        --out_dir "$REPO_ROOT/outputs/q2_${ARM}_k${K}"
fi
