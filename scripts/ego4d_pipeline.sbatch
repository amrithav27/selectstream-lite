#!/bin/bash
# Whole chain in one 4-GPU allocation on one node (docs/plan.md §4). Data (clips + model) are copied to the
# node by ego4d_env.sh if it does not hold them; the feature cache, checkpoints and traces stay in its scratch,
# so resubmit to the same node (-w) to resume. Each finished stage writes outputs/<RUN>/done.<stage>.
#   0 cache (4 shards in parallel, one GPU each) -> 0b verify against the online path
#   1 scorer (Stage 1 + Check 1) -> 1b write-side replay (Check 2, CPU, runs alongside Stage R)
#   2 Stage R (arm A) -> 3 calibrate (all train-fit streams) -> 4 Stage W + eval per arm (A, B first)
#   5 Q2 (both tracks, arms A and B) -> 6 memory budget N (eval-time)
# Pilot (every code path on a few streams; caches only their clips):
#   RUN=pilot5 PILOT=1 sbatch --time=0-06:00:00 -w <node> scripts/ego4d_pipeline.sbatch
# Full run (all streams; empty sizes = all):
#   RUN=full sbatch -w <node> scripts/ego4d_pipeline.sbatch
#SBATCH --job-name=ss_pipe
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G               # = 16 CPUs at 3000 MB/CPU (4 cache shards x 3 decode workers; Stage R + replay ~31 GB)
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail  # no -e: one failed stage is logged and its dependents are skipped, the rest still run
source scripts/ego4d_env.sh || exit 1
RUN="${RUN:?set RUN (tag for this run)}"
if [ "${PILOT:-0}" = 1 ]; then
    : "${R_STREAMS:=4}" "${R_EPOCHS:=1}" "${CAL_STREAMS:=4}" "${W_STEPS:=1}" "${EVAL_STREAMS:=2}" "${Q2_STREAMS:=2}"
    : "${N_ABL:=64}" "${N_ABL_STREAMS:=1}" "${SCORER_STREAMS:=6}" "${SCORER_SEEDS:=1}" "${SCORER_DIAG:=}" "${REPLAY_STREAMS:=6}"
fi
R_STREAMS="${R_STREAMS:-}"            # empty = all train-fit streams
R_EPOCHS="${R_EPOCHS:-2}"
CAL_STREAMS="${CAL_STREAMS:-}"
W_STEPS="${W_STEPS:-50}"              # Stage W per arm: optimizer steps x 4 streams, equal for every arm
EVAL_STREAMS="${EVAL_STREAMS:-}"      # empty = all 300 val streams
Q2_STREAMS="${Q2_STREAMS:-}"
N_ABL="${N_ABL:-64 128}"
N_ABL_MIN_FRAMES="${N_ABL_MIN_FRAMES:-1536}"
N_ABL_STREAMS="${N_ABL_STREAMS:-}"
SCORER_STREAMS="${SCORER_STREAMS:-}"
SCORER_SEEDS="${SCORER_SEEDS:-3}"
SCORER_DIAG="${SCORER_DIAG-128}"
REPLAY_STREAMS="${REPLAY_STREAMS:-}"
opt() { [ -n "$2" ] && echo "$1 $2"; }  # opt --max_clips "$X" -> nothing when X is empty

RUN_DIR="$REPO_ROOT/outputs/$RUN"                 # small files: logs, summaries, figures, markers
CK="${CK_DIR:-$CKPT_HOME/$RUN}"                   # weights (1.6 GB each)
OUT="$OUT_ROOT/$RUN"                              # node scratch: resume state, traces
CACHE="${CACHE_DIR:-$SCRATCH/feature_cache}"
SCORER="$RUN_DIR/scorer/scorer.pt"
mkdir -p "$RUN_DIR" "$CK" "$OUT" "$CACHE"
PROBE_CAL="$REPO_ROOT/outputs/calibration.json"   # pre-Stage-R probe (job 17646): tau_s, tau_r for arm A
CAL="$RUN_DIR/calibration.json"
echo "[pipe] RUN=$RUN PILOT=${PILOT:-0} R=${R_STREAMS:-all}x${R_EPOCHS}ep CAL=${CAL_STREAMS:-all} W_STEPS=$W_STEPS EVAL=${EVAL_STREAMS:-all} Q2=${Q2_STREAMS:-all} N_ABL=$N_ABL"
"$PY" -c "import torch; print('[pipe] GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

FAILED=0
done_() { [ -f "$RUN_DIR/done.$1" ]; }
mark() { touch "$RUN_DIR/done.$1"; echo "[pipe] ===== $1 done $(date +%F_%T)"; }
start() { echo "[pipe] ===== $1 start $(date +%F_%T)"; }

# ---- 0. Feature cache (shared by every run on this node; resumable per clip) ----
CLIPS_FILE=""
if [ "${PILOT:-0}" = 1 ]; then
    CLIPS_FILE="$RUN_DIR/pilot_clips.txt"
    "$PY" - "$CLIPS_FILE" "$SCORER_STREAMS" <<'EOF'
import sys
from streamlite.data.ego4d_qa import Ego4DStreamDataset
out, n = sys.argv[1], int(sys.argv[2])
uids = []
for split, src, part in (("train", "merged", "train"), ("train", "merged", "dev"), ("val", "nlq", "all"), ("val", "qaego4d", "all")):
    d = Ego4DStreamDataset("dataset/clean", "", split, src, part=part, require_video=False)
    for it in d.items[:n]:
        uids += it.clip_uids
open(out, "w").write("\n".join(dict.fromkeys(uids)) + "\n")
print(f"[pipe] pilot clips: {len(set(uids))}")
EOF
fi
report() { "$PY" -m streamlite.cli.plot_ego4d --run_dir "$RUN_DIR" > "$RUN_DIR/plot.log" 2>&1; tail -1 "$RUN_DIR/plot.log"; }

# Clips: a node outside NODES that got no copy from a home node fetches the missing ones from Drive.
if [ -n "$CLIPS_FILE" ]; then
    missing=$(while read -r u; do [ -f "$VIDEO_DIR/$u.mp4" ] || echo "$u"; done < "$CLIPS_FILE" | wc -l)
    [ "$missing" -gt 0 ] && { echo "[pipe] $missing pilot clips missing: staging them from Drive"; stage_clips "$CLIPS_FILE"; }
else
    have=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
    if [ "$have" -lt "$EXPECTED_CLIPS" ]; then stage_clips; else echo "[pipe] $have clips already in $VIDEO_DIR"; fi
fi

if ! done_ cache; then
    start cache
    for i in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$i "$PY" -m streamlite.cli.cache_features --video_dir "$VIDEO_DIR" --out_dir "$CACHE" --shard $i --num_shards 4 \
            --workers 3 $(opt --clips_file "$CLIPS_FILE") > "$RUN_DIR/cache_shard$i.log" 2>&1 &
    done
    wait
    tail -n 2 "$RUN_DIR"/cache_shard*.log
    n_ok=$(grep -l '"frame_format": "png", "error": null' "$CACHE"/meta/*.json 2>/dev/null | wc -l); n_err=$(grep -l '"error": "' "$CACHE"/meta/*.json 2>/dev/null | wc -l)
    echo "[pipe] cache: $n_ok clips ok, $n_err with errors ($(du -sh "$CACHE" | cut -f1))"
    grep -h '"error": "' "$CACHE"/meta/*.json 2>/dev/null | head -5
    [ "$n_ok" -gt 0 ] && mark cache || exit 1
fi
find "$CACHE" -maxdepth 1 -exec touch {} +  # keep the 7-day scratch eviction away from the directories

if ! done_ verify; then
    start verify
    "$PY" -m streamlite.cli.verify_cache --video_dir "$VIDEO_DIR" --cache_dir "$CACHE" --num_clips 2 --stride 40 --out "$RUN_DIR/verify_cache.json" \
        && mark verify || { echo "[pipe][ERROR] cache does not match the online path; stopping"; exit 1; }
fi

# ---- 1. Stage 1: scorer + Check 1 ----
if ! done_ scorer; then
    start scorer
    "$PY" -m streamlite.cli.train_scorer --feature_cache "$CACHE" --out_dir "$RUN_DIR/scorer" --seeds "$SCORER_SEEDS" \
        --diag_proj_dims "$SCORER_DIAG" $(opt --max_streams "$SCORER_STREAMS") && mark scorer || exit 1
    report
fi

# ---- 1b. Check 2 (CPU) runs in the background next to Stage R ----
REPLAY_PID=""
if ! done_ replay; then
    start replay
    ( "$PY" -m streamlite.cli.replay_write --feature_cache "$CACHE" --scorer "$SCORER" --out_dir "$RUN_DIR/replay" \
        $(opt --max_streams "$REPLAY_STREAMS") > "$RUN_DIR/replay.log" 2>&1 && mark replay ) &
    REPLAY_PID=$!
fi

train() {  # train <stage> <arm> <extra args...>
    local st=$1 arm=$2; shift 2
    local name="${st}_${arm}" o="$OUT/${st}_${arm}"
    done_ "$name" && return 0
    start "$name"
    "$PY" -m streamlite.cli.train_ego4d --stage "$st" --arm "$arm" --video_dir "$VIDEO_DIR" --output_dir "$o" --feature_cache "$CACHE" \
        --scorer "$SCORER" --trace_dir "$o/traces" --trace_clips 2 --save_every_clips 10 "$@" || { echo "[pipe][ERROR] $name failed"; FAILED=$((FAILED + 1)); return 1; }
    mkdir -p "$CK/$name" "$RUN_DIR/$name"
    cp "$o/final/weights.pt" "$CK/$name/weights.pt" && cp "$o/train_log.jsonl" "$o/config.json" "$RUN_DIR/$name/"
    "$PY" -m streamlite.cli.plot_ego4d --traces "$o/traces" --train_log "$o/train_log.jsonl" \
        --config configs/ego4d_qwen25vl7b.yaml --out "$RUN_DIR/$name/figs" || true
    mark "$name"
}

evaluate() {  # evaluate <name> <arm> <ckpt|none> <track> <extra args...>
    local name=$1 arm=$2 ck=$3 track=$4; shift 4
    local o="$OUT/$name"
    done_ "$name" && return 0
    start "$name"
    "$PY" -m streamlite.cli.eval_ego4d --arm "$arm" --ckpt "$ck" --track "$track" --video_dir "$VIDEO_DIR" --feature_cache "$CACHE" \
        --scorer "$SCORER" --out_dir "$o" --trace_dir "$o/traces" --trace_clips 2 --calibration "$CAL" "$@" || { echo "[pipe][ERROR] $name failed"; FAILED=$((FAILED + 1)); return 1; }
    mkdir -p "$RUN_DIR/$name"
    cp "$o/summary.json" "$o/predictions.jsonl" "$RUN_DIR/$name/"
    "$PY" -m streamlite.cli.plot_ego4d --traces "$o/traces" --summaries "$o/summary.json" --out "$RUN_DIR/$name/figs" || true
    mark "$name"
}

# ---- 2. Stage R: reader + writer, arm A, thresholds from the pre-Stage-R probe ----
# SEG_CENTER=train_mean subtracts the train mean g_t before the segment encoder (none = the paper; docs/plan.md §5).
train R A --epochs "$R_EPOCHS" $(opt --max_clips "$R_STREAMS") --calibration "$PROBE_CAL" --segment_center "${SEG_CENTER:-train_mean}" || exit 1
[ -f "$OUT/R_A/segment_center.npy" ] && cp "$OUT/R_A/segment_center.npy" "$CK/R_A/"

# ---- 3. Calibrate arms B/C/D against arm A's segment count (trained writer, Stage-1 scorer) ----
if ! done_ calibrate; then
    start calibrate
    "$PY" -m streamlite.cli.calibrate_write --ckpt "$CK/R_A" --scorer "$SCORER" --feature_cache "$CACHE" --video_dir "$VIDEO_DIR" \
        $(opt --num_clips "$CAL_STREAMS") --out "$CAL" > "$RUN_DIR/calibrate.log" 2>&1 && mark calibrate || { tail -20 "$RUN_DIR/calibrate.log"; exit 1; }
    tail -40 "$RUN_DIR/calibrate.log"
fi

# ---- 4. Stage W per arm (writer only, equal steps, same stream order) and both tracks ----
wtrain_eval() {
    local arm=$1
    train W "$arm" --init_from "$CK/R_A" --max_steps "$W_STEPS" --calibration "$CAL" || return 1
    evaluate "eval_qa_$arm" "$arm" "$CK/W_$arm" qa $(opt --max_clips "$EVAL_STREAMS")
    evaluate "eval_nlq_$arm" "$arm" "$CK/W_$arm" nlq $(opt --max_clips "$EVAL_STREAMS")
}
wtrain_eval A
wtrain_eval B
report
# Same streams and ask times, trainables at their seed-42 init: did training change anything?
evaluate eval_qa_untrained A none qa $(opt --max_clips "$EVAL_STREAMS") $(opt --segment_center "$([ -f "$CK/R_A/segment_center.npy" ] && echo "$CK/R_A/segment_center.npy")")
wtrain_eval C
wtrain_eval D
# Arm B with the un-retrained Stage-R writer: what Stage W buys.
evaluate eval_qa_B_Rwriter B "$CK/R_A" qa $(opt --max_clips "$EVAL_STREAMS")

# ---- 5. Q2 on both tracks, arms A and B ----
for arm in A B; do
    for track in nlq qa; do
        name="q2_${track}_$arm"
        if [ -f "$CK/W_$arm/weights.pt" ] && ! done_ "$name"; then
            start "$name"
            "$PY" -m streamlite.cli.eval_q2 --arm "$arm" --ckpt "$CK/W_$arm" --scorer "$SCORER" --track "$track" --ks 2,3 \
                --video_dir "$VIDEO_DIR" --feature_cache "$CACHE" --calibration "$CAL" --out_dir "$RUN_DIR/$name" \
                $(opt --max_clips "$Q2_STREAMS") && mark "$name" || { echo "[pipe][ERROR] $name failed"; FAILED=$((FAILED + 1)); }
        fi
    done
done

# ---- 6. Memory budget N, eval-time only (trained at N = 16), on the longest val streams ----
if [ -f "$CK/W_A/weights.pt" ]; then
    for n in 16 $N_ABL; do
        evaluate "eval_qa_A_N$n" A "$CK/W_A" qa --num_slots "$n" --min_frames "$N_ABL_MIN_FRAMES" $(opt --max_clips "$N_ABL_STREAMS")
    done
fi
[ -n "$REPLAY_PID" ] && { wait "$REPLAY_PID" || { echo "[pipe][ERROR] replay failed (see replay.log)"; FAILED=$((FAILED + 1)); }; }
report
echo "[pipe] finished $(date +%F_%T) with $FAILED failed stage(s)"; ls "$RUN_DIR"
exit $(( FAILED > 0 ))
