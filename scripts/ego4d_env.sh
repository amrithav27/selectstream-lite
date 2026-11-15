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
# The login token lives in the default ~/.cache/huggingface, not in HF_HOME on scratch: without this, private
# Hub repos return 401 from compute nodes (job 74554 failed on it).
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
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

check_deps() {
    # RF-10b: an interactive `hf auth login` upgraded huggingface_hub to 2.x under a running pipeline, and
    # transformers 5.5.4 refuses to import with it. Fail fast at job start instead of mid-pipeline.
    "$PY" - <<'EOF' || { echo "[env][ERROR] dependency check failed (see above); fix the venv before running"; return 1; }
import sys
from importlib.metadata import version
hub, tf = version("huggingface_hub"), version("transformers")
print(f"[env] huggingface_hub {hub}, transformers {tf}")
if int(hub.split(".")[0]) >= 2:
    sys.exit(f"[env][ERROR] huggingface_hub {hub} >= 2.0 breaks transformers {tf}; pip install --no-deps 'huggingface_hub==1.11.0'")
import transformers  # noqa: F401  (the import itself is the check)
EOF
}

safe_copy() {  # safe_copy <src> <dst> [sha256 record]: copy, verify byte-for-byte, record the hash; delete on mismatch.
    local src=$1 dst=$2 rec=${3:-}
    if ! cp "$src" "$dst.tmp" || ! cmp -s "$src" "$dst.tmp"; then
        rm -f "$dst.tmp"; echo "[env][ERROR] copy $src -> $dst failed or differs (disk full?)"; return 1
    fi
    mv "$dst.tmp" "$dst" || return 1
    [ -n "$rec" ] && sha256sum "$dst" | awk -v p="$(basename "$dst")" '{print $1"  "p}' > "$rec"
    return 0
}

fetch_ckpt() {  # fetch_ckpt <run> <stage_arm> <dest dir>: Hub copy of a checkpoint when it is not on this node.
    local run=$1 name=$2 dest=$3
    [ -f "$dest/weights.pt" ] && return 0
    mkdir -p "$dest"
    echo "[env] fetching checkpoints/$run/$name from the Hub into $dest"
    HF_HUB_OFFLINE=0 "$PY" - "$run" "$name" "$dest" <<'EOF' || { echo "[env][ERROR] could not fetch $run/$name"; return 1; }
import os, shutil, sys
from huggingface_hub import hf_hub_download
run, name, dest = sys.argv[1:]
repo = os.environ.get("HF_CKPT_REPO", "amritha27/selectstream-lite-ego4d")
for f in ("weights.pt", "segment_center.npy"):
    try:
        p = hf_hub_download(repo, f"checkpoints/{run}/{name}/{f}", local_dir=os.path.join(dest, ".hub"))
    except Exception as e:
        if f == "weights.pt":
            raise
        continue
    shutil.move(p, os.path.join(dest, f))
print(f"[env] {dest}: {sorted(os.listdir(dest))}")
EOF
}

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
