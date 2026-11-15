#!/bin/bash
# Runs that settle the open red flags of a finished pipeline run (docs/red_flags.md). Same node as the run
# (its scratch holds the feature cache and the per-stage outputs):
#   RUN=full_v3 sbatch -w gnode047 scripts/ego4d_redflags.sbatch
# Steps (each resumable through outputs/<RUN>/done.<name>):
#   0  checkpoints: verify scratch weights against the Hub copy (sha256); W_B latest vs final (RF-11)
#   1  read controls, 150 val QA streams, paired with read_full (RF-1, RF-15, RF-16, P1 text-only)
#   2  seeds 43, 44 for Stage W arms A and B + both evals (RF-6 / AB-R1)
#   3  memory budget N = 64, 128 on all val QA streams, paired with eval_qa_A at N = 16 (RF-18)
#   4  write-side grid rerun into ablate_write_rerun/ (RF-19), CPU, in the background
#   5  scripts/collect_results.py: every comparison with CI + permutation p
#SBATCH --job-name=ss_redflags
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

set -uo pipefail  # no -e: a failed step is counted and the next one still runs
trap 'echo "[rf] exit code $? at $(date +%F_%T)"' EXIT
source scripts/ego4d_env.sh || exit 1
check_deps || exit 1
RUN="${RUN:?set RUN}"
READ_STREAMS="${READ_STREAMS:-150}"
SEEDS="${SEEDS:-43 44}"
W_STEPS="${W_STEPS:-50}"
RUN_DIR="$REPO_ROOT/outputs/$RUN"
OUT="$OUT_ROOT/$RUN"                        # the pipeline's scratch outputs (final/ and latest/ per stage)
CK="$SCRATCH/ckpt/$RUN"                     # verified weights used by this job
CACHE="${CACHE_DIR:-$SCRATCH/feature_cache}"
SCORER="$RUN_DIR/scorer/scorer.pt"
CAL="$RUN_DIR/calibration.json"
mkdir -p "$CK"
"$PY" -c "import torch; print('[rf] GPUs:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
FAILED=0
done_() { [ -f "$RUN_DIR/done.$1" ]; }
mark() { touch "$RUN_DIR/done.$1"; echo "[rf] ===== $1 done $(date +%F_%T)"; }
start() { echo "[rf] ===== $1 start $(date +%F_%T) (scratch free: $(df -h --output=avail /ssd_scratch | tail -1))"; }

# ---- 0. Checkpoints: scratch final/ weights must equal the Hub copy the reported results used ----
start ckpt_verify
HF_HUB_OFFLINE=0 "$PY" - "$RUN" "$OUT" "$CK" "$RUN_DIR/ckpt_verify.json" <<'EOF' || { echo "[rf][ERROR] checkpoint verification failed"; exit 1; }
import hashlib, json, os, sys
import torch
from huggingface_hub import HfApi
run, out, ck, rec = sys.argv[1:]
repo = os.environ.get("HF_CKPT_REPO", "amritha27/selectstream-lite-ego4d")
names = ["R_A", "W_A", "W_B", "W_C", "W_D"]
info = {p.path: p for p in HfApi().get_paths_info(repo, [f"checkpoints/{run}/{n}/weights.pt" for n in names])}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
res = {}
for n in names:
    src = os.path.join(out, n, "final", "weights.pt")
    hub = info.get(f"checkpoints/{run}/{n}/weights.pt")
    r = {"scratch": src, "exists": os.path.exists(src), "hub_sha256": hub.lfs.sha256 if hub and hub.lfs else None}
    if r["exists"]:
        r["sha256"] = sha(src)
        r["match"] = r["sha256"] == r["hub_sha256"]
        if r["match"]:
            os.makedirs(os.path.join(ck, n), exist_ok=True)
            dst = os.path.join(ck, n, "weights.pt")
            if not os.path.exists(dst):
                os.link(src, dst)  # same filesystem: no extra space
            seg = os.path.join(out, n, "segment_center.npy")
            if os.path.exists(seg) and not os.path.exists(os.path.join(ck, n, "segment_center.npy")):
                os.link(seg, os.path.join(ck, n, "segment_center.npy"))
    res[n] = r
    print(f"[rf] {n}: scratch={'yes' if r['exists'] else 'no'} sha_match={r.get('match')}")
# RF-11: W_B was resumed from latest/trainable.pt of the failed job. Do its model tensors equal final/weights.pt?
lt, fw = os.path.join(out, "W_B", "latest", "trainable.pt"), os.path.join(out, "W_B", "final", "weights.pt")
if os.path.exists(lt) and os.path.exists(fw):
    full = torch.load(lt, map_location="cpu", weights_only=False)
    a = full.get("model", full)
    b = torch.load(fw, map_location="cpu", weights_only=False)
    b = b.get("model", b)
    keys = sorted(set(a) & set(b))
    diff = [k for k in keys if not torch.equal(a[k], b[k])]
    res["RF11_W_B_latest_vs_final"] = {"keys_common": len(keys), "only_latest": len(set(a) - set(b)), "only_final": len(set(b) - set(a)),
                                       "differing": diff[:20], "n_differing": len(diff), "step_latest": full.get("step") if isinstance(full, dict) else None}
    print("[rf] RF-11:", res["RF11_W_B_latest_vs_final"])
json.dump(res, open(rec, "w"), indent=1)
bad = [n for n in ("R_A", "W_A", "W_B") if not res[n].get("match")]
if bad:
    sys.exit(f"[rf][ERROR] no verified scratch weights for {bad}; fetch them with fetch_ckpt before rerunning")
EOF
mark ckpt_verify

evaluate() {  # evaluate <name> <arm> <ckpt> <track> <extra args...>
    local name=$1 arm=$2 ck=$3 track=$4; shift 4
    local o="$OUT/$name"
    done_ "$name" && return 0
    start "$name"
    "$PY" -m streamlite.cli.eval_ego4d --arm "$arm" --ckpt "$ck" --track "$track" --video_dir "$VIDEO_DIR" --feature_cache "$CACHE" \
        --scorer "$SCORER" --calibration "$CAL" --out_dir "$o" --trace_clips 0 "$@" \
        || { echo "[rf][ERROR] $name failed"; FAILED=$((FAILED + 1)); return 1; }
    mkdir -p "$RUN_DIR/$name"
    cp "$o/summary.json" "$o/predictions.jsonl" "$RUN_DIR/$name/" || { echo "[rf][ERROR] $name: copy failed"; FAILED=$((FAILED + 1)); return 1; }
    mark "$name"
}

# ---- 1. Read controls (W_A, first READ_STREAMS val QA streams; pair with read_full) ----
R="--max_clips $READ_STREAMS"
evaluate read_text_only     A "$CK/W_A" qa $R --evidence_k 0 --blank_frame        # P1: question text only
evaluate read_zero_evidence A "$CK/W_A" qa $R --read_ablation zero_evidence       # RF-1: tokens present, content zero
evaluate read_noise_evidence A "$CK/W_A" qa $R --read_ablation noise_evidence     # RF-1: tokens present, content noise
evaluate read_oracle_wrong  A "$CK/W_A" qa $R --read_ablation oracle_wrong        # RF-1: real nodes, all outside the window
evaluate read_no_mark_read  A "$CK/W_A" qa $R --no_mark_read                       # RF-16: reads do not change keep/evict
evaluate read_seed_topk8    A "$CK/W_A" qa $R --seed_topk 8                        # RF-15: re-rank chooses 4 of 8 seeds
evaluate read_seed_topk8_coarse A "$CK/W_A" qa $R --seed_topk 8 --read_ablation coarse_only

# ---- 4 (background, CPU). Write-side grid rerun (RF-19): fresh dir, so it is a reproduction, not a copy ----
WRITE_PID=""
if ! done_ ablate_write_rerun; then
    ( RUN="$RUN" POOLED="$SCORER" OUT="$RUN_DIR/ablate_write_rerun" CACHE="$CACHE" WORKERS=6 \
        bash scripts/ego4d_ablate.sh > "$RUN_DIR/ablate_write_rerun.log" 2>&1 && mark ablate_write_rerun ) &
    WRITE_PID=$!
fi

# ---- 2. Seeds for Stage W, arms A and B (RF-6). Scratch is tight: keep only final/weights.pt ----
# Order is by value per GPU-hour (submission deadline 2026-11-15): the first extra seed gives B − A a second
# seed; the budget runs come next; the second extra seed last.
seed_runs() {
    local s=$1 arm name o free_gb
    for arm in A B; do
        name="W_${arm}_seed$s"; o="$OUT/$name"
        if ! done_ "$name"; then
            # Job 75224: scratch hit 0 B mid-save and left a truncated epoch0/ (other users share this disk). A
            # Stage W run peaks at ~12 GB (latest + epoch0 + its temp copy + final), so check first and start clean.
            free_gb=$(df -BG --output=avail /ssd_scratch | tail -1 | tr -dc 0-9)
            if [ "$free_gb" -lt "${MIN_TRAIN_GB:-20}" ]; then
                echo "[rf][ERROR] $name skipped: ${free_gb} GB free in /ssd_scratch, need ${MIN_TRAIN_GB:-20}"; FAILED=$((FAILED + 1)); continue
            fi
            rm -rf "$o"
            start "$name"
            "$PY" -m streamlite.cli.train_ego4d --stage W --arm "$arm" --video_dir "$VIDEO_DIR" --output_dir "$o" --feature_cache "$CACHE" \
                --scorer "$SCORER" --init_from "$CK/R_A" --max_steps "$W_STEPS" --calibration "$CAL" --seed "$s" \
                --trace_clips 0 --save_every_clips 10 \
                && mkdir -p "$RUN_DIR/$name" && cp "$o/train_log.jsonl" "$o/config.json" "$RUN_DIR/$name/" \
                && sha256sum "$o/final/weights.pt" > "$RUN_DIR/$name/weights.sha256" && rm -rf "$o/latest" "$o"/epoch* && mark "$name" \
                || { echo "[rf][ERROR] $name failed"; FAILED=$((FAILED + 1)); continue; }
        fi
        evaluate "eval_qa_${arm}_seed$s"  "$arm" "$o" qa
        evaluate "eval_nlq_${arm}_seed$s" "$arm" "$o" nlq
    done
}

SEED_LIST=($SEEDS)
seed_runs "${SEED_LIST[0]}"

# ---- 3. Memory budget on all val QA streams (RF-18); N = 16 is eval_qa_A ----
for n in 64 128; do
    evaluate "budget_qa_A_N$n" A "$CK/W_A" qa --num_slots "$n"
done

for s in "${SEED_LIST[@]:1}"; do seed_runs "$s"; done

if [ -n "$WRITE_PID" ]; then
    wait "$WRITE_PID" || { echo "[rf][ERROR] write-side rerun failed: $(tail -2 "$RUN_DIR/ablate_write_rerun.log")"; FAILED=$((FAILED + 1)); }
fi

# ---- 5. Every comparison, with a test ----
"$PY" scripts/collect_results.py "$RUN_DIR" "$RUN_DIR/collected_results.json" || FAILED=$((FAILED + 1))
echo "[rf] finished $(date +%F_%T) with $FAILED failed step(s)"
exit $((FAILED > 0))
