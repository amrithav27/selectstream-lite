#!/bin/bash
# Copy staged clips from this node's /ssd_scratch to another node's, over the cluster network
# (faster than re-downloading from Drive). Run on the SOURCE node; ssh into DEST works only while we
# have a job running there (pam_slurm_adopt). Add-only: --ignore-existing, no --delete.
# WITH_MODEL=1 also copies the Qwen cache (hf_cache, ~16 GB, with its completeness marker).
#   DEST=gnode044 sbatch -w gnode061 scripts/ego4d_copy_clips.sbatch
# ego4d_env.sh submits this automatically when a job lands on a node without the data.
#SBATCH --job-name=ss_copy
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
DEST="${DEST:?set DEST=<node>}"
SRC_DIR="/ssd_scratch/$USER/selectstream/clips/"
DST_DIR="/ssd_scratch/$USER/selectstream/clips/"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
echo "[copy] $(hostname -s) -> $DEST  src=$(ls "$SRC_DIR" | grep -c '\.mp4$') mp4  dest_before=$($SSH "$DEST" "ls $DST_DIR | grep -c '\.mp4\$'") mp4"

$SSH "$DEST" "mkdir -p $DST_DIR"
rsync -a --ignore-existing --include='*.mp4' --exclude='*' --bwlimit=100000 -e "$SSH" --stats "$SRC_DIR" "$DEST:$DST_DIR" | tail -15
STATUS=${PIPESTATUS[0]}
if [ "${WITH_MODEL:-0}" = 1 ]; then
    HF="/ssd_scratch/$USER/selectstream/hf_cache/"
    # The completeness marker goes last, and only after the cache itself copied without error.
    if rsync -a --ignore-existing --exclude='.qwen25vl7b_complete' --bwlimit=100000 -e "$SSH" "$HF" "$DEST:$HF"; then
        rsync -a -e "$SSH" "$HF.qwen25vl7b_complete" "$DEST:$HF" && echo "[copy] model cache copied"
    else
        echo "[copy][ERROR] model cache copy failed (marker not copied, DEST will download)"; STATUS=1
    fi
fi

# Size check: files present in both with different sizes, and files still missing on DEST.
ls -l "$SRC_DIR" | awk '/\.mp4$/{print $NF, $5}' | sort > "/tmp/ss_copy_src_$SLURM_JOB_ID"
$SSH "$DEST" "ls -l $DST_DIR" | awk '/\.mp4$/{print $NF, $5}' | sort > "/tmp/ss_copy_dst_$SLURM_JOB_ID"
echo "[copy] rsync exit=$STATUS  dest_after=$(wc -l < /tmp/ss_copy_dst_$SLURM_JOB_ID) mp4  " \
     "missing_or_size_mismatch=$(comm -23 /tmp/ss_copy_src_$SLURM_JOB_ID /tmp/ss_copy_dst_$SLURM_JOB_ID | wc -l)"
comm -23 "/tmp/ss_copy_src_$SLURM_JOB_ID" "/tmp/ss_copy_dst_$SLURM_JOB_ID" | head -5
rm -f "/tmp/ss_copy_src_$SLURM_JOB_ID" "/tmp/ss_copy_dst_$SLURM_JOB_ID"
exit $STATUS
