#!/bin/bash
# Upload the Stage-0 feature cache's g/ and meta/ (not the PNG frames) to a private Hugging Face dataset repo.
# Must run on the node that holds the cache (node-local /ssd_scratch), as a CPU-only job:
#   sbatch -w gnode047 --account=research --partition=u22 --qos=medium --cpus-per-task=2 --mem=6G \
#       --time=06:00:00 --output=logs/%x-%j.out --job-name=ss_hffeat \
#       --export=ALL,HF_REPO=amritha27/selectstream-lite-ego4d-features scripts/hf_upload_features.sh
# One commit per subfolder; rerunning skips files already on the Hub (same hash).
set -euo pipefail
HF_REPO="${HF_REPO:?set HF_REPO}"
PY="${PY:-$HOME/miniforge3/envs/venv/bin/python}"
CACHE="${CACHE_DIR:-/ssd_scratch/$USER/selectstream/feature_cache}"
[ -d "$CACHE/g" ] && [ -d "$CACHE/meta" ] || { echo "[hf] no cache at $CACHE (wrong node?)"; exit 1; }
echo "[hf] node=$(hostname) cache=$CACHE: g $(ls "$CACHE/g" | wc -l) files, meta $(ls "$CACHE/meta" | wc -l) files"
"$PY" - "$HF_REPO" "$CACHE" <<'EOF'
import sys
from huggingface_hub import HfApi
repo, cache = sys.argv[1], sys.argv[2]
api = HfApi()
print("[hf] as", api.whoami()["name"], flush=True)
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
for sub in ("meta", "g"):
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=f"{cache}/{sub}", path_in_repo=sub,
                      commit_message=f"Add feature cache {sub}/")
    print(f"[hf] {sub}/ uploaded", flush=True)
files = api.list_repo_files(repo, repo_type="dataset")
print(f"[hf] done: https://huggingface.co/datasets/{repo} ({len(files)} files)")
EOF
