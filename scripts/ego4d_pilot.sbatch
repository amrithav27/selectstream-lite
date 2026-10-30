#!/bin/bash
# Pilot (docs/pilot.md): 10 train streams, Stage R, arm A. DTYPES / EPOCHS / TAG override the defaults.
# Pilot 1 (2026-10-10) ran DTYPES="bfloat16 float16" EPOCHS=3 on single clips.
# Measures s/clip, peak GPU memory, and whether fp16 visual features overflow (the model raises
# FloatingPointError on inf/NaN). Writes traces + figures for 3 clips.
#   sbatch -w gnode061 scripts/ego4d_pilot.sbatch
#SBATCH --job-name=ss_pilot
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
source scripts/ego4d_env.sh || exit 1

"$PY" - <<'EOF' > "$SCRATCH/pilot_clips.txt"
import json, random
clips = [c for c in json.load(open("dataset/clean/clips.json")) if c["split"] == "train" and c["drive_file_id"]]
random.Random(42).shuffle(clips)
print("\n".join(c["clip_uid"] for c in clips[:40]))
EOF
# Streams = whole videos: stage every train clip of those videos, not only the listed ones.
"$PY" - "$SCRATCH/pilot_clips.txt" <<'PYEOF' > "$SCRATCH/pilot_stream_clips.txt"
import json, sys
clips = json.load(open("dataset/clean/clips.json"))
listed = set(open(sys.argv[1]).read().split())
vids = {c["video_uid"] for c in clips if c["clip_uid"] in listed}
print("\n".join(c["clip_uid"] for c in clips if c["video_uid"] in vids and c["split"] == "train"))
PYEOF
stage_clips "$SCRATCH/pilot_stream_clips.txt" || exit 1

for DT in ${DTYPES:-float16}; do
    OUT="$OUT_ROOT/${TAG:-pilot2}_${DT}"
    rm -rf "$OUT"
    echo "===== pilot dtype=$DT ====="
    "$PY" -m streamlite.cli.train_ego4d --stage R --arm A --torch_dtype "$DT" \
        --video_dir "$VIDEO_DIR" --output_dir "$OUT" --clips_file "$SCRATCH/pilot_clips.txt" --max_clips 10 --epochs "${EPOCHS:-1}" \
        --trace_dir "$OUT/traces" --trace_clips 3 --save_every_clips 1000 \
        || echo "[pilot] dtype=$DT FAILED (see stderr)"
    "$PY" -m streamlite.cli.plot_ego4d --traces "$OUT/traces" --train_log "$OUT/train_log.jsonl" \
        --config configs/ego4d_qwen25vl7b.yaml --out "$REPO_ROOT/outputs/${TAG:-pilot2}_${DT}/figs" || true
    mkdir -p "$REPO_ROOT/outputs/${TAG:-pilot2}_${DT}"
    cp "$OUT/train_log.jsonl" "$OUT/config.json" "$REPO_ROOT/outputs/${TAG:-pilot2}_${DT}/" 2>/dev/null || true
done
echo "[pilot] done"
