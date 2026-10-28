#!/bin/bash
# Mirror every clip on Drive into gnode061's /ssd_scratch (node-local, so it must run on the same node
# as training). No GPU: copying needs none, and the GPUs stay free for training jobs.
# --ignore-existing: safe to re-run, and safe alongside a job staging a subset (rclone writes to a
# temp name and renames). Does not source ego4d_env.sh, so it never touches the model download.
#   sbatch -w gnode061 scripts/ego4d_stage_all.sbatch
#SBATCH --job-name=ss_stage
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
export PATH="$HOME/bin:$PATH"  # rclone: only ada-gw1 has it installed; ~/bin/rclone is a copy of /usr/bin/rclone
VIDEO_DIR="/ssd_scratch/$USER/selectstream/clips"
SRC="${RCLONE_REMOTE:-gdrive}:${DRIVE_CLIPS_PATH:-ego4d_nlq/clips}"
mkdir -p "$VIDEO_DIR"
echo "[stage] node=$(hostname -s) job=${SLURM_JOB_ID:-none} src=$SRC dst=$VIDEO_DIR"
df -h /ssd_scratch | tail -1
echo "[stage] on Drive: $(rclone lsf "$SRC" --files-only --bind 0.0.0.0 | wc -l) files, $(rclone size "$SRC" --bind 0.0.0.0 | tr '\n' ' ')"
echo "[stage] already staged: $(find "$VIDEO_DIR" -name '*.mp4' | wc -l)"

rclone copy "$SRC" "$VIDEO_DIR" --ignore-existing --transfers 8 --checkers 16 --bind 0.0.0.0 \
    --stats 5m --stats-one-line -v > "$VIDEO_DIR/../stage_all.rclone.log" 2>&1
STATUS=$?
grep -E "Transferred|ERROR|error" "$VIDEO_DIR/../stage_all.rclone.log" | tail -20

find "$VIDEO_DIR" -type f -exec touch {} +
echo "[stage] rclone exit=$STATUS; staged: $(find "$VIDEO_DIR" -name '*.mp4' | wc -l) mp4, $(du -sh "$VIDEO_DIR" | cut -f1)"
rclone check "$SRC" "$VIDEO_DIR" --one-way --size-only --bind 0.0.0.0 2>&1 | tail -3
exit $STATUS
