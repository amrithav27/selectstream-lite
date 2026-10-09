#!/bin/bash
# Single-node job: pull the 100 visualiser clips from Drive into /ssd_scratch,
# then run a quantised Qwen-VL model via Ollama to fill the annotation gaps
# the dataset visualiser flags as "needs_vlm_annotation" (num_people,
# semantic_location, country-relevant visual cues).
#
# Kept as ONE job on ONE node on purpose: /ssd_scratch is node-local and
# evicted after 7 days of no access, so splitting download and inference
# across separate jobs risks landing on different nodes and re-downloading.
# Re-running this same script later reuses whatever's already in
# /ssd_scratch/$USER/ego4d_nlq_100 if it lands on the same node; the rclone
# and Ollama install steps are both idempotent (skip what's already there).
#
#SBATCH --job-name=qwen_vl_annotate
#SBATCH --account=research
#SBATCH --partition=long
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail  # no -e: later independent steps (per-clip annotation) must not die on one failure

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/logs"

# ---- config -----------------------------------------------------------
RCLONE_REMOTE="${RCLONE_REMOTE:-gdrive}"             # set up once: `rclone config` on ada-gw1
DRIVE_CLIPS_PATH="${DRIVE_CLIPS_PATH:-ego4d_nlq/clips}"
SCRATCH_ROOT="/ssd_scratch/$USER/ego4d_nlq_100"
MANIFEST="$REPO_ROOT/dataset/viz/manifest.json"
QWEN_MODEL="${QWEN_MODEL:-qwen2.5vl:32b}"             # best Qwen-VL that fits 4x11GB at q4_K_M; drop to :7b if OOM
FRAMES_PER_CLIP="${FRAMES_PER_CLIP:-6}"

echo "[env] node=$(hostname) GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ---- 1. stage clips from Drive to /ssd_scratch -------------------------
mkdir -p "$SCRATCH_ROOT/clips" "$SCRATCH_ROOT/annotations"

python3 -I - "$MANIFEST" <<'PYEOF' > "$SCRATCH_ROOT/clip_uids.txt"
import json, sys
m = json.load(open(sys.argv[1]))
uids = {q["clip_uid"] for v in m["videos"] for q in v["nlq_queries"]}
print("\n".join(sorted(uids)))
PYEOF

echo "[rclone] staging $(wc -l < "$SCRATCH_ROOT/clip_uids.txt") clips"
while read -r clip_uid; do
    dest="$SCRATCH_ROOT/clips/${clip_uid}.mp4"
    [ -f "$dest" ] && continue   # already staged (same node as a previous run) -- skip, don't re-pull
    rclone copy "${RCLONE_REMOTE}:${DRIVE_CLIPS_PATH}/${clip_uid}.mp4" "$SCRATCH_ROOT/clips/" \
        -q || echo "[rclone][WARN] missing on Drive: $clip_uid"
done < "$SCRATCH_ROOT/clip_uids.txt"

# touch everything so the 7-day scratch eviction clock resets on this run
find "$SCRATCH_ROOT/clips" -type f -exec touch {} +

# ---- 2. install Ollama fresh into /ssd_scratch (per existing workflow) --
OLLAMA_ROOT="/ssd_scratch/$USER/ollama_install"
OLLAMA_DIR="$OLLAMA_ROOT/usr"

if [ -f "$OLLAMA_DIR/bin/ollama" ]; then
    echo "[Ollama] Already installed at $OLLAMA_DIR"
else
    mkdir -p "$OLLAMA_DIR" "$OLLAMA_ROOT/models"
    if [ ! -f ~/ollama ]; then
        echo "[ERROR] ~/ollama tarball not found. rsync it first."
        exit 1
    fi
    cp ~/ollama "$OLLAMA_ROOT/"
    tar -C "$OLLAMA_DIR" -xf "$OLLAMA_ROOT/ollama"
fi

export HF_HOME=/ssd_scratch/$USER/hf_cache
export OLLAMA_MODELS="$OLLAMA_ROOT/models"
export PYTHONUNBUFFERED=1
export OLLAMA_BASE_URL="http://localhost:11434"
export PATH="$OLLAMA_DIR/bin:$PATH"

"$OLLAMA_DIR/bin/ollama" serve > "$REPO_ROOT/logs/ollama_server_${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

for i in $(seq 1 60); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[Ollama] Server ready after ${i}s"
        break
    fi
    sleep 1
done
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "[ERROR] Ollama server failed to start."
    kill "$OLLAMA_PID" 2>/dev/null || true
    exit 1
fi

echo "[Ollama] pulling $QWEN_MODEL (standard GGUF quant -- not forcing bf16, see docs/ for why)"
"$OLLAMA_DIR/bin/ollama" pull "$QWEN_MODEL"

# ---- 3. run annotation over the 100 videos' clips ----------------------
python3 "$REPO_ROOT/scripts/qwen_annotate.py" \
    --manifest "$MANIFEST" \
    --clips-dir "$SCRATCH_ROOT/clips" \
    --out "$SCRATCH_ROOT/annotations/vlm_annotations.json" \
    --model "$QWEN_MODEL" \
    --frames-per-clip "$FRAMES_PER_CLIP"

"$OLLAMA_DIR/bin/ollama" stop "$QWEN_MODEL"
kill "$OLLAMA_PID" 2>/dev/null || true

# ---- 4. bring results back through the gateway (compute nodes can't see /share1 or home2 NFS well) --
cp "$SCRATCH_ROOT/annotations/vlm_annotations.json" "$REPO_ROOT/dataset/viz/vlm_annotations.json"
echo "[done] wrote $REPO_ROOT/dataset/viz/vlm_annotations.json"
