#!/bin/bash
# Submit an ego4d job to the node where it can start soonest. Run from the repo root on ada-gw1:
#   STAGE=R ARM=A bash scripts/ego4d_submit.sh scripts/ego4d_train.sbatch
#   bash scripts/ego4d_submit.sh --dependency=afterok:123 scripts/ego4d_eval.sbatch
# Candidates: the home nodes (NODES, which hold all clips and the model) and every other 2080 Ti
# node that is up. `sbatch --test-only` gives each one's estimated start. A home node wins unless
# another node starts more than PREFER_HOME_S (default 2 h) earlier; a job on another node fetches
# the data from a home node first (scripts/ego4d_env.sh). Prints the chosen node and the job id.
# DRY_RUN=1 only reports the choice.
set -uo pipefail
cd "$(dirname "$0")/.."
NODES="${NODES:-gnode061 gnode044}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gnode079}"  # gnode079: 17 GB scratch free (2026-10-12)
PREFER_HOME_S="${PREFER_HOME_S:-7200}"

start_of() {  # start_of <node> <sbatch args...> -> epoch seconds of the estimated start, or nothing
    local node=$1; shift
    local out ts
    out=$(sbatch --test-only -w "$node" "$@" 2>&1)
    ts=$(echo "$out" | grep -oE 'start at [0-9T:-]+' | awk '{print $3}')
    [ -n "$ts" ] && date -d "$ts" +%s
}

others=$(sinfo -N -h -p u22 -o "%N %t %f" | awk '$3 ~ /2080ti/ && $2 !~ /(drain|down|fail|maint)/ {print $1}' | sort -u)
best_home="" best_home_s="" best_other="" best_other_s=""
for n in $NODES; do
    s=$(start_of "$n" "$@") || continue
    [ -z "$s" ] && continue
    if [ -z "$best_home" ] || [ "$s" -lt "$best_home_s" ]; then best_home=$n; best_home_s=$s; fi
done
for n in $others; do
    case " $NODES $EXCLUDE_NODES " in *" $n "*) continue ;; esac
    s=$(start_of "$n" "$@") || continue
    [ -z "$s" ] && continue
    if [ -z "$best_other" ] || [ "$s" -lt "$best_other_s" ]; then best_other=$n; best_other_s=$s; fi
done

fmt() { [ -n "$1" ] && date -d "@$1" "+%F %H:%M" || echo "n/a"; }
echo "[submit] best home node: ${best_home:-none} at $(fmt "$best_home_s"); best other: ${best_other:-none} at $(fmt "$best_other_s")" >&2
if [ -n "$best_home" ] && { [ -z "$best_other" ] || [ "$best_home_s" -le $((best_other_s + PREFER_HOME_S)) ]; }; then
    node=$best_home
elif [ -n "$best_other" ]; then
    node=$best_other
else
    echo "[submit][ERROR] no candidate node accepted the job (sbatch --test-only)" >&2
    exit 1
fi
echo "[submit] -> $node" >&2
if [ -n "${DRY_RUN:-}" ]; then echo "[submit] DRY_RUN: not submitted" >&2; exit 0; fi
sbatch --parsable -w "$node" "$@"
