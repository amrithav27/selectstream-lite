#!/bin/bash
# Evaluate one arm on one track (never both tracks in one stream; plan §1), then plot.
#   ARM=A CKPT=$OUT_ROOT/R_A/epoch2 TRACK=qa  sbatch -w gnode061 scripts/ego4d_eval.sbatch
#   ARM=A CKPT=... TRACK=nlq SPLIT=train (= dev participants, early stopping) sbatch -w gnode061 ...
#   SKIP_STAGE=1 MAX_CLIPS=5 ...: smoke eval on clips already staged (loader drops clips without an mp4)
#SBATCH --job-name=ss_eval
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
source scripts/ego4d_env.sh || exit 1
ARM="${ARM:-A}"
CKPT="${CKPT:?set CKPT}"
TRACK="${TRACK:?set TRACK=qa or nlq}"
SPLIT="${SPLIT:-val}"
TAG="eval_${SPLIT}_${TRACK}_${ARM}_$(basename "$(dirname "$CKPT")")_$(basename "$CKPT")"
OUT="$OUT_ROOT/$TAG"
EXTRA=()
[ -n "${MAX_CLIPS:-}" ] && EXTRA+=(--max_clips "$MAX_CLIPS")

if [ "${SKIP_STAGE:-0}" = 1 ]; then echo "[eval] SKIP_STAGE=1: using clips already in $VIDEO_DIR"; else stage_clips || exit 1; fi
"$PY" -m streamlite.cli.eval_ego4d --arm "$ARM" --ckpt "$CKPT" --track "$TRACK" --split "$SPLIT" \
    --video_dir "$VIDEO_DIR" --out_dir "$OUT" --trace_dir "$OUT/traces" --trace_clips 3 "${EXTRA[@]}"
STATUS=$?
mkdir -p "$REPO_ROOT/outputs/$TAG"
cp "$OUT/summary.json" "$OUT/predictions.jsonl" "$REPO_ROOT/outputs/$TAG/" 2>/dev/null || true
"$PY" -m streamlite.cli.plot_ego4d --traces "$OUT/traces" --summaries "$OUT/summary.json" --out "$REPO_ROOT/outputs/$TAG/figs" || true
exit $STATUS
