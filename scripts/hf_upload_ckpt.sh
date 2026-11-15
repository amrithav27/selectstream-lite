#!/bin/bash
# Upload checkpoints and every run's outputs to the Hugging Face model repo, in this layout:
#   checkpoints/<RUN>/<stage>/weights.pt (+ R_A/segment_center.npy)   trained modules per stage
#   checkpoints/full/R_A/weights.pt                                    the paper's reader as written (run 43319)
#   runs/<run>/...                                                     outputs/<run> (evals, logs, scorer, calibration, figures, ablations)
#   runs/full_ablate_offline/, runs/pilots/<run>/, runs/diagnostics/   offline ablations, pilots, probe scripts
#   config/, README.md, ABLATIONS.md
# Stages hardlinks in ~/hf_stage (no extra space on /home2) and uploads with upload_folder: files already on the Hub
# are skipped by hash, so rerun it after more stages finish. Old top-level paths from the first upload are deleted.
# Needs > 2 GB RAM (it hashes 1.7 GB files), so run it as a CPU job, not on the gateway:
#   sbatch --account=research --partition=u22 --qos=medium --cpus-per-task=2 --mem=6G --time=06:00:00 \
#       --output=logs/%x-%j.out --job-name=ss_hfup --export=ALL,RUN=full_v3,HF_REPO=amritha27/selectstream-lite-ego4d \
#       scripts/hf_upload_ckpt.sh
set -euo pipefail
RUN="${RUN:?set RUN}"
HF_REPO="${HF_REPO:?set HF_REPO}"
PY="${PY:-$HOME/miniforge3/envs/venv/bin/python}"
REPO_ROOT="${REPO_ROOT:-$HOME/selectstream-lite}"
ABL_ROOT="${ABL_ROOT:-$HOME/selectstream-abl}"
CKPT_HOME="${CKPT_HOME:-$HOME/selectstream_ckpt}"
STAGE="$HOME/hf_stage/all"

rm -rf "$STAGE" && mkdir -p "$STAGE"
link() { mkdir -p "$(dirname "$STAGE/$2")" && ln "$1" "$STAGE/$2"; }
linkdir() {  # linkdir <src dir> <dest prefix>: every file except done markers
    (cd "$1" && find . -type f ! -name 'done.*' -printf '%P\n') | while read -r f; do link "$1/$f" "$2/$f"; done
}

# Checkpoints: only stages with a done marker, so a truncated copy never goes up.
for d in "$CKPT_HOME/$RUN"/*/; do
    name=$(basename "$d")
    [ -f "$d/weights.pt" ] && [ -f "$REPO_ROOT/outputs/$RUN/done.$name" ] || { echo "[hf] skip $RUN/$name"; continue; }
    link "$d/weights.pt" "checkpoints/$RUN/$name/weights.pt"
    [ -f "$d/segment_center.npy" ] && link "$d/segment_center.npy" "checkpoints/$RUN/$name/segment_center.npy"
done
[ -f "$CKPT_HOME/full/R_A/weights.pt" ] && link "$CKPT_HOME/full/R_A/weights.pt" checkpoints/full/R_A/weights.pt

# Run outputs.
for r in full_v3 full full_v2; do [ -d "$REPO_ROOT/outputs/$r" ] && linkdir "$REPO_ROOT/outputs/$r" "runs/$r"; done
for r in pilot_bfloat16 pilot_float16 pilot2_float16 pilot3_float16 eval_val_qa_A_pilot3_float16_final \
         eval_val_nlq_A_pilot3_float16_final pilot5 q2_A_k2; do
    [ -d "$REPO_ROOT/outputs/$r" ] && linkdir "$REPO_ROOT/outputs/$r" "runs/pilots/$r"
done
[ -d "$ABL_ROOT/outputs/full_ablate" ] && linkdir "$ABL_ROOT/outputs/full_ablate" runs/full_ablate_offline
[ -f "$ABL_ROOT/outputs/full_ablate.log" ] && link "$ABL_ROOT/outputs/full_ablate.log" runs/full_ablate_offline/full_ablate.log
for f in zeroshot_probe.py zeroshot_nodes.py aniso.py; do
    [ -f "$ABL_ROOT/outputs/$f" ] && link "$ABL_ROOT/outputs/$f" "runs/diagnostics/$f"
done

link "$REPO_ROOT/configs/ego4d_qwen25vl7b.yaml" config/ego4d_qwen25vl7b.yaml
cp "$REPO_ROOT/docs/hf_README.md" "$STAGE/README.md"
cp "$REPO_ROOT/docs/ablations_for_team.md" "$STAGE/ABLATIONS.md"

echo "[hf] staged $(find "$STAGE" -type f | wc -l) files, $(du -shL "$STAGE" | cut -f1)"
(cd "$STAGE" && find . -type f -path './checkpoints/*' | sort)
"$PY" - "$HF_REPO" "$STAGE" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, folder = sys.argv[1], sys.argv[2]
api = HfApi()
print("[hf] as", api.whoami()["name"], flush=True)
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
# RF-20: never pass delete_patterns. A pattern matches nested paths and deleted checkpoints/full/R_A on 2026-10-30.
api.upload_folder(repo_id=repo, repo_type="model", folder_path=folder,
                  commit_message="Add W_D, Q2, memory-budget and read ablations")
files = api.list_repo_files(repo)
print(f"[hf] done: https://huggingface.co/{repo} ({len(files)} files)")
EOF
