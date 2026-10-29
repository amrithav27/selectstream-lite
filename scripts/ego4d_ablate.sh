#!/bin/bash
# Offline write-side ablations (docs/plan.md §13.3), CPU only. Run on a node that holds the feature cache,
# e.g. inside a pipeline allocation: ssh <node> 'cd <repo> && bash scripts/ego4d_ablate.sh'.
#   1. scorer variants (train_scorer.py)  -> $OUT/scorers/<name>/scorer.pt + check1.json
#   2. replay grid (ablate_write.py)      -> $OUT/ablate_raw, $OUT/ablate_norm (value signal rank-normalised)
#   3. figures (plot_ego4d.py --run_dir)  -> $OUT/figures/ablate_*.png
# Each step is skipped when its output exists, so a rerun resumes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PY="${PY:-$HOME/miniforge3/envs/venv/bin/python}"
RUN="${RUN:-full}"
CACHE="${CACHE:-/ssd_scratch/$USER/selectstream/feature_cache}"
OUT="${OUT:-$REPO_ROOT/outputs/$RUN}"
POOLED="${POOLED:-$OUT/scorer/scorer.pt}"  # the Stage-1 scorer the pipeline trained (arm B)
WORKERS="${WORKERS:-6}"
L_MINS="${L_MINS:-8,4,2}"
BUDGETS="${BUDGETS:-8,16,32}"
MAX_STREAMS="${MAX_STREAMS:-}"  # smoke tests only: first N streams of every split
SMALL=${MAX_STREAMS:+--max_streams $MAX_STREAMS}
export CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_ROOT" PYTHONUNBUFFERED=1
mkdir -p "$OUT/scorers"
FAILED=0

# name | train_scorer flags. One seed each (the pooled scorer has three); selection by dev within-video AUC
# except for bce, which keeps Stage 1's dev AP.
SCORERS=(
    "stream|--loss stream_bce"
    "rank|--loss rank_local"
    "gonly|--loss stream_bce --input_mask g"
    "scalars|--loss stream_bce --input_mask scalars"
    "tol2|--loss stream_bce --time_tolerance 2"
    "shift|--loss stream_bce --label_shift"
    "pooled40|--loss bce --max_epochs 40 --patience 5"
)
for spec in "${SCORERS[@]}"; do
    name=${spec%%|*}; flags=${spec#*|}
    d="$OUT/scorers/$name"
    [ -f "$d/scorer.pt" ] && { echo "[ablate] scorer $name exists"; continue; }
    echo "[ablate] scorer $name: $flags ($(date +%H:%M))"
    nice -n 10 "$PY" -m main.cli.train_scorer --feature_cache "$CACHE" --out_dir "$d" --seeds 1 --diag_proj_dims "" $SMALL $flags \
        > "$d.log" 2>&1 || { echo "[ablate][ERROR] scorer $name failed: $(tail -2 "$d.log")"; FAILED=$((FAILED + 1)); }
done

SC="pooled=$POOLED"
for spec in "${SCORERS[@]}"; do
    name=${spec%%|*}
    [ -f "$OUT/scorers/$name/scorer.pt" ] && SC="$SC,$name=$OUT/scorers/$name/scorer.pt"
done
B=$(echo "$SC" | tr ',' '\n' | cut -d= -f1 | sed 's/^/B./' | paste -sd,)
# AB-G2 (L_min x core signals), AB-G1 (c4), AB-I1/L1/L2 (scorer files), AB-I2 (E), AB-U1 (cut:value splits).
VARIANTS="A,As,C,Cr,D,c4,E,$B,A:B.pooled,B.pooled:A,A:B.stream,B.stream:A,C:A,A:C"

run_grid() {  # run_grid <dir> <extra flags>
    local d="$OUT/$1"; shift
    [ -f "$d/ablate_write.json" ] && { echo "[ablate] $d exists"; return; }
    mkdir -p "$d"
    echo "[ablate] grid $d ($(date +%H:%M))"
    nice -n 10 "$PY" -m main.cli.ablate_write --feature_cache "$CACHE" --out_dir "$d" --scorers "$SC" --variants "$VARIANTS" \
        --l_mins "$L_MINS" --budgets "$BUDGETS" --workers "$WORKERS" $SMALL "$@" > "$d/ablate.log" 2>&1 \
        || { echo "[ablate][ERROR] grid $d failed: $(tail -2 "$d/ablate.log")"; FAILED=$((FAILED + 1)); }
}
run_grid ablate_raw
run_grid ablate_norm --norm_value

"$PY" -m main.cli.plot_ego4d --run_dir "$OUT" > "$OUT/plot_ablate.log" 2>&1; tail -1 "$OUT/plot_ablate.log"
echo "[ablate] finished $(date +%H:%M) with $FAILED failed step(s)"
exit $((FAILED > 0))
