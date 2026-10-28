#!/bin/bash
# Train one stage/arm. Resumes from latest/ after preemption or the wall limit: re-submit as-is, on any node.
#   STAGE=R ARM=A EPOCHS=1                                  bash scripts/ego4d_submit.sh scripts/ego4d_train.sbatch
#   STAGE=W ARM=B INIT_FROM=~/selectstream_ckpt/R_A MAX_STEPS=300 bash scripts/ego4d_submit.sh scripts/ego4d_train.sbatch
# latest/ (weights + optimizer + RNG, 4.9 GB) is mirrored to $CKPT_HOME/<stage>_<arm>/ when the job ends or
# 15 min before the wall limit (SIGUSR1), and restored from there when this node's copy is older or missing.
#SBATCH --job-name=ss_train
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --signal=B:USR1@900

set -uo pipefail
source scripts/ego4d_env.sh || exit 1
STAGE="${STAGE:?set STAGE=R or W}"
ARM="${ARM:-A}"
OUT="$OUT_ROOT/${STAGE}_${ARM}"
EXTRA=()
[ -n "${INIT_FROM:-}" ] && EXTRA+=(--init_from "$INIT_FROM")
[ -n "${MAX_STEPS:-}" ] && EXTRA+=(--max_steps "$MAX_STEPS")
[ -n "${MAX_CLIPS:-}" ] && EXTRA+=(--max_clips "$MAX_CLIPS")
[ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
HOME_RUN="$CKPT_HOME/${STAGE}_${ARM}"

mirror_latest() {  # node scratch -> /home (atomic: copy, then rename)
    [ -f "$OUT/latest/trainable.pt" ] || return 0
    mkdir -p "$HOME_RUN/latest"
    cp "$OUT/latest/trainable.pt" "$HOME_RUN/latest/trainable.pt.tmp" && mv "$HOME_RUN/latest/trainable.pt.tmp" "$HOME_RUN/latest/trainable.pt"
    cp "$OUT/train_log.jsonl" "$HOME_RUN/train_log.jsonl" 2>/dev/null || true
    echo "[train] mirrored latest/ to $HOME_RUN ($(du -sh "$CKPT_HOME" | cut -f1) in $CKPT_HOME)"
}
# Resume state from another node: take /home's copy when it is newer than this node's (or this has none).
if [ -f "$HOME_RUN/latest/trainable.pt" ] && { [ ! -f "$OUT/latest/trainable.pt" ] || [ "$HOME_RUN/latest/trainable.pt" -nt "$OUT/latest/trainable.pt" ]; }; then
    mkdir -p "$OUT/latest" && cp "$HOME_RUN/latest/trainable.pt" "$OUT/latest/trainable.pt"
    cp "$HOME_RUN/train_log.jsonl" "$OUT/train_log.jsonl" 2>/dev/null || true
    echo "[train] restored latest/ from $HOME_RUN"
fi
trap 'echo "[train] wall limit near (USR1)"; mirror_latest' USR1

stage_clips || exit 1
# Run in the background and wait, so the USR1 trap can fire while Python runs.
"$PY" -m main.cli.train_ego4d --stage "$STAGE" --arm "$ARM" --video_dir "$VIDEO_DIR" --output_dir "$OUT" \
    --trace_dir "$OUT/traces" --trace_clips 3 "${EXTRA[@]}" &
PID=$!
wait $PID; STATUS=$?
while kill -0 $PID 2>/dev/null; do wait $PID; STATUS=$?; done
mirror_latest

# Weights-only copy on /home so the next stage can run on either node.
if [ -f "$OUT/final/weights.pt" ]; then
    mkdir -p "$CKPT_HOME/${STAGE}_${ARM}" && cp "$OUT/final/weights.pt" "$CKPT_HOME/${STAGE}_${ARM}/weights.pt"
    echo "[train] mirrored final weights to $CKPT_HOME/${STAGE}_${ARM}/ ($(du -sh "$CKPT_HOME" | cut -f1) in $CKPT_HOME)"
fi
# A finished stage needs only weights.pt on /home; drop the 4.9 GB resume copy to save /home quota.
if [ "$STATUS" = 0 ] && [ -f "$OUT/final/weights.pt" ] && [ -z "${KEEP_HOME_LATEST:-}" ]; then
    rm -rf "$HOME_RUN/latest"
fi
mkdir -p "$REPO_ROOT/outputs/${STAGE}_${ARM}"
cp "$OUT/train_log.jsonl" "$REPO_ROOT/outputs/${STAGE}_${ARM}/" 2>/dev/null || true
"$PY" -m main.cli.plot_ego4d --traces "$OUT/traces" --train_log "$OUT/train_log.jsonl" \
    --config configs/ego4d_qwen25vl7b.yaml --out "$REPO_ROOT/outputs/${STAGE}_${ARM}/figs" || true
exit $STATUS
