# Sourced by every scripts/ego4d_*.sbatch. Two home nodes (NODES) hold all clips and the model in their
# node-local /ssd_scratch (evicted after 7 days without access, so every job touches the files). A job
# that lands elsewhere (scripts/ego4d_submit.sh picks the earliest free 2080 Ti node) first has a home
# node push the data to it over the cluster network; Drive / HF are the fallback. Checkpoints later
# stages need are mirrored to $CKPT_HOME on /home, which every node sees.
export NODES="${NODES:-gnode061 gnode044}"
export NODE="${NODE:-${NODES%% *}}"
export CKPT_HOME="${CKPT_HOME:-$HOME/selectstream_ckpt}"
export SCRATCH="/ssd_scratch/$USER/selectstream"
export VIDEO_DIR="$SCRATCH/clips"
export OUT_ROOT="$SCRATCH/outputs"
export PY="$HOME/miniforge3/envs/venv/bin/python"
export PATH="$HOME/bin:$PATH"  # rclone: only ada-gw1 has it installed; ~/bin/rclone is a copy of /usr/bin/rclone
export RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"
export DRIVE_CLIPS_PATH="${DRIVE_CLIPS_PATH:-ego4d_nlq/clips}"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs" "$VIDEO_DIR" "$OUT_ROOT" "$HF_HOME/hub"

mkdir -p "$CKPT_HOME"
echo "[env] node=$(hostname -s) job=${SLURM_JOB_ID:-none} repo=$REPO_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
df -h /ssd_scratch | tail -1

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_MARKER="$HF_HOME/.qwen25vl7b_complete"
EXPECTED_CLIPS=$("$PY" -c "import json; print(len(json.load(open('dataset/clean/clips.json'))))")

fetch_from_home_node() {
    # On a node outside NODES: submit scripts/ego4d_copy_clips.sbatch on the home node whose copy job
    # can start soonest (it pushes clips + model here; ssh into this node works because this job runs
    # here) and wait for it. Gives up after FETCH_MAX_WAIT_S of queueing, leaving Drive / HF to fill in.
    local host have free_gb best="" best_start=0 n out start_s job state waited=0
    host=$(hostname -s)
    have=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
    if [ "$have" -ge "$EXPECTED_CLIPS" ] && [ -f "$MODEL_MARKER" ]; then return 0; fi
    free_gb=$(df -BG --output=avail /ssd_scratch | tail -1 | tr -dc 0-9)
    if [ "$free_gb" -lt "${MIN_FREE_GB:-200}" ]; then
        echo "[env][ERROR] $host: ${free_gb} GB free in /ssd_scratch, need ${MIN_FREE_GB:-200} for clips + model + checkpoints"
        return 1
    fi
    for n in $NODES; do
        out=$(DEST=$host WITH_MODEL=1 sbatch --test-only -w "$n" scripts/ego4d_copy_clips.sbatch 2>&1)
        start_s=$(date -d "$(echo "$out" | grep -oE 'start at [0-9T:-]+' | awk '{print $3}')" +%s 2>/dev/null) || continue
        if [ -z "$best" ] || [ "$start_s" -lt "$best_start" ]; then best=$n; best_start=$start_s; fi
    done
    if [ -z "$best" ] || [ $(( best_start - $(date +%s) )) -gt "${FETCH_MAX_WAIT_S:-1800}" ]; then
        echo "[env][WARN] no home node can start a copy job within ${FETCH_MAX_WAIT_S:-1800}s; falling back to Drive / HF"
        return 0
    fi
    job=$(DEST=$host WITH_MODEL=1 sbatch --parsable -w "$best" scripts/ego4d_copy_clips.sbatch) || return 0
    echo "[env] $host has $have/$EXPECTED_CLIPS clips: copying from $best (job $job)"
    while state=$(squeue -h -j "$job" -o %T 2>/dev/null) && [ -n "$state" ]; do
        if [ "$state" = PENDING ] && [ "$waited" -gt "${FETCH_MAX_WAIT_S:-1800}" ]; then
            scancel "$job"; echo "[env][WARN] copy job $job still pending; cancelled, falling back to Drive / HF"; return 0
        fi
        sleep 30; waited=$((waited + 30))
    done
    grep -h "^\[copy\]" "logs/ss_copy-$job.out" 2>/dev/null
}

case " $NODES " in
    *" $(hostname -s) "*) ;;
    *) fetch_from_home_node || { return 1 2>/dev/null || exit 1; } ;;
esac

# Model weights: once per node in scratch (~16 GB; too big for /home next to everything else). The
# marker is written only after a complete download/copy, so an interrupted job retries; later jobs on
# the same node reuse it with HF_HUB_OFFLINE=1.
if [ ! -f "$MODEL_MARKER" ]; then
    echo "[env] downloading $MODEL_ID to $HF_HOME"
    HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL_ID'))" \
        && touch "$MODEL_MARKER" || { echo "[env][ERROR] model download failed"; return 1 2>/dev/null || exit 1; }
fi
find "$HF_HOME" -type f -exec touch {} +  # keep the 7-day scratch eviction away

stage_clips() {
    # Usage: stage_clips [file-with-clip-uids]. Without a list, mirrors the whole Drive folder.
    # --ignore-existing makes re-runs only fetch new clips (e.g. the ones still downloading to Drive).
    if ! command -v rclone >/dev/null; then
        echo "[stage][ERROR] rclone not found on $(hostname -s); expected ~/bin/rclone (copy of ada-gw1's /usr/bin/rclone)."
        return 1
    fi
    if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:$"; then
        echo "[stage][ERROR] rclone remote '${RCLONE_REMOTE}' is not configured on Ada. See docs/plan.md §5."
        return 1
    fi
    local args=(copy "${RCLONE_REMOTE}:${DRIVE_CLIPS_PATH}" "$VIDEO_DIR" --ignore-existing --transfers 8 --checkers 16 --bind 0.0.0.0 -q)
    if [ -n "${1:-}" ]; then
        sed 's/$/.mp4/' "$1" > "$SCRATCH/files_from.txt"
        args+=(--files-from "$SCRATCH/files_from.txt")
    fi
    rclone "${args[@]}" || echo "[stage][WARN] rclone exited non-zero; missing clips are skipped by the loader"
    find "$VIDEO_DIR" -type f -exec touch {} +
    echo "[stage] $(find "$VIDEO_DIR" -name '*.mp4' | wc -l) clips in $VIDEO_DIR ($(du -sh "$VIDEO_DIR" | cut -f1))"
}
