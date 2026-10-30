#!/bin/bash
# Read-side (retrieval) and query-conditioning ablations (docs/plan.md §13.5), after a pipeline run. Every variant
# evaluates the run's trained arm-A checkpoint (W_A) on the same first READ_STREAMS val QA streams, so rows pair
# by query with the "full" reference. Then the CPU write-side and offline read-side ablations, then figures.
#   RUN=full_v3 sbatch -w gnode047 --dependency=afterany:<pipeline job> scripts/ego4d_ablate_read.sbatch
# Outputs: outputs/<RUN>/read_<variant>/{summary.json,predictions.jsonl}, outputs/<RUN>/ablate_read/,
# outputs/<RUN>/ablate_{raw,norm}/, outputs/<RUN>/figures/{read_*,ablate_*}.png. Resumable (done markers).
#SBATCH --job-name=ss_ablate
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
source scripts/ego4d_env.sh || exit 1
RUN="${RUN:?set RUN (the pipeline run to ablate)}"
READ_STREAMS="${READ_STREAMS:-150}"   # of 243 val QA streams; ~12 questions each
RUN_DIR="$REPO_ROOT/outputs/$RUN"
CK="${CK_DIR:-$CKPT_HOME/$RUN}"
OUT="$OUT_ROOT/$RUN"
CACHE="${CACHE_DIR:-$SCRATCH/feature_cache}"
SCORER="$RUN_DIR/scorer/scorer.pt"
CAL="$RUN_DIR/calibration.json"
W_A="$CK/W_A"
[ -f "$W_A/weights.pt" ] || { echo "[ablate][ERROR] $W_A/weights.pt missing: the pipeline run did not finish Stage W for arm A"; exit 1; }
FAILED=0
done_() { [ -f "$RUN_DIR/done.$1" ]; }
mark() { touch "$RUN_DIR/done.$1"; echo "[ablate] ===== $1 done $(date +%F_%T)"; }

read_eval() {  # read_eval <name> <eval flags...>
    local name="read_$1"; shift
    done_ "$name" && return 0
    echo "[ablate] ===== $name start $(date +%F_%T)"
    "$PY" -m streamlite.cli.eval_ego4d --arm A --ckpt "$W_A" --track qa --video_dir "$VIDEO_DIR" --feature_cache "$CACHE" --scorer "$SCORER" \
        --calibration "$CAL" --out_dir "$OUT/$name" --max_clips "$READ_STREAMS" "$@" \
        || { echo "[ablate][ERROR] $name failed"; FAILED=$((FAILED + 1)); return 1; }
    mkdir -p "$RUN_DIR/$name" && cp "$OUT/$name/summary.json" "$OUT/$name/predictions.jsonl" "$RUN_DIR/$name/"
    mark "$name"
}

# CPU write-side grid (unchanged by the reader; rerun so every ablation of this run sits in its outputs) runs in the
# background next to the GPU evals.
WRITE_PID=""
if ! done_ ablate_write; then
    ( RUN="$RUN" POOLED="$SCORER" OUT="$RUN_DIR" bash scripts/ego4d_ablate.sh > "$RUN_DIR/ablate_write.log" 2>&1 && mark ablate_write ) &
    WRITE_PID=$!
fi

# Reference, then by question: which Eq. 6 terms rank (score); does the graph help (hops, GAR, refined order);
# bounds (random, oracle); how much evidence (M; 0 = no memory); the open segment; query conditioning.
read_eval full
for ab in no_prior no_cos no_bias cos_only prior_only; do read_eval "score_$ab" --read_ablation "$ab"; done
for ab in no_hops no_gar coarse_only; do read_eval "graph_$ab" --read_ablation "$ab"; done
for ab in random oracle; do read_eval "bound_$ab" --read_ablation "$ab"; done
for m in 0 1 2 8; do read_eval "M$m" --evidence_k "$m"; done
read_eval no_open --no_open
read_eval blank_frame --blank_frame

# CPU: offline read grid (training-free scores).
if ! done_ ablate_read; then
    "$PY" -m streamlite.cli.ablate_read --feature_cache "$CACHE" --check2 "$RUN_DIR/replay/check2.json" \
        --segment_center "$CK/R_A/segment_center.npy" --out_dir "$RUN_DIR/ablate_read" > "$RUN_DIR/ablate_read.log" 2>&1 \
        && mark ablate_read || { echo "[ablate][ERROR] ablate_read failed: $(tail -2 "$RUN_DIR/ablate_read.log")"; FAILED=$((FAILED + 1)); }
fi
if [ -n "$WRITE_PID" ]; then
    wait "$WRITE_PID" || { echo "[ablate][ERROR] write-side grid failed: $(tail -2 "$RUN_DIR/ablate_write.log")"; FAILED=$((FAILED + 1)); }
fi

"$PY" -m streamlite.cli.plot_ego4d --run_dir "$RUN_DIR" > "$RUN_DIR/plot_ablate.log" 2>&1; tail -1 "$RUN_DIR/plot_ablate.log"
echo "[ablate] finished $(date +%F_%T) with $FAILED failed step(s)"
exit $((FAILED > 0))
