#!/bin/bash
# Stage 0: cache g_t + 1-fps frames for one shard of the clips (main/cli/cache_features.py). One GPU.
# Four shards fill the QoS (4 GPUs) and run side by side on one node:
#   for i in 0 1 2 3; do SHARD=$i NUM_SHARDS=4 sbatch -w gnode061 scripts/ego4d_cache.sbatch; done
# Pilot: SHARD=0 NUM_SHARDS=1 MAX_CLIPS=2 sbatch -w gnode061 --time=0-01:00:00 scripts/ego4d_cache.sbatch
# Resumable: finished clips are skipped. Output: $SCRATCH/feature_cache (node-local).
#SBATCH --job-name=ss_cache
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G               # = 8 CPUs at 3000 MB/CPU
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
source scripts/ego4d_env.sh || exit 1
"$PY" -c "import torch; print('[cache] GPU:', torch.cuda.get_device_name(0))"
CACHE_DIR="${CACHE_DIR:-$SCRATCH/feature_cache}"
mkdir -p "$CACHE_DIR"
"$PY" -m main.cli.cache_features --video_dir "$VIDEO_DIR" --out_dir "$CACHE_DIR" \
    --shard "${SHARD:-0}" --num_shards "${NUM_SHARDS:-1}" --splits "${SPLITS:-train,val,test}" \
    --workers "${WORKERS:-6}" ${MAX_CLIPS:+--max_clips "$MAX_CLIPS"}
STATUS=$?
echo "[cache] $(ls "$CACHE_DIR/meta" 2>/dev/null | wc -l) clips in $CACHE_DIR ($(du -sh "$CACHE_DIR" | cut -f1)); errors: $(grep -l '"error": "' "$CACHE_DIR"/meta/*.json 2>/dev/null | wc -l)"
exit $STATUS
