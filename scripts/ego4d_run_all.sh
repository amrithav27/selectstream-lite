#!/bin/bash
# Submit the whole chain (docs/plan.md §5) on one node, each job waiting for the previous with afterok:
#   Stage R (arm A) -> calibrate -> Stage W x {A,B,C,D} -> eval {qa,nlq} x arms + Q2 (arm A).
# Run from the repo root on ada-gw1, after the pilot:   W_STEPS=300 bash scripts/ego4d_run_all.sh
# Skip finished stages with R_JOB=done / CAL_JOB=done. If a job hits the 4-day wall it fails, its
# dependents stay pending (DependencyNeverSatisfied): scancel them, re-submit the same stage (it
# resumes from latest/), then re-run this script with that stage marked done.
set -uo pipefail
cd "$(dirname "$0")/.."
NODE="${NODE:-auto}"  # auto: each job goes where it can start soonest (scripts/ego4d_submit.sh); or a fixed node
W_STEPS="${W_STEPS:?set W_STEPS = optimizer steps per Stage-W arm (equal for every arm)}"

CKPT_HOME="${CKPT_HOME:-$HOME/selectstream_ckpt}"  # weights mirrored here by ego4d_train.sbatch
R_CKPT="$CKPT_HOME/R_A"

submit() {  # submit <dependency-jobid|done|""> <env...> -- <script>
    local dep="$1"; shift
    local envs=()
    while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
    local dflag=()
    [ -n "$dep" ] && [ "$dep" != "done" ] && dflag=(--dependency="afterok:$dep")
    if [ "$NODE" = auto ]; then
        env "${envs[@]}" bash scripts/ego4d_submit.sh "${dflag[@]}" "$@"
    else
        env "${envs[@]}" sbatch --parsable -w "$NODE" "${dflag[@]}" "$@"
    fi
}

R_JOB="${R_JOB:-$(submit "" STAGE=R ARM=A -- scripts/ego4d_train.sbatch)}"
echo "stage R: $R_JOB  (ckpt $R_CKPT)"
CAL_JOB="${CAL_JOB:-$(submit "$R_JOB" TASK=calibrate CKPT="$R_CKPT" -- scripts/ego4d_calibrate_q2.sbatch)}"
echo "calibrate: $CAL_JOB"
for ARM in A B C D; do
    W=$(submit "$CAL_JOB" STAGE=W ARM=$ARM INIT_FROM="$R_CKPT" MAX_STEPS="$W_STEPS" -- scripts/ego4d_train.sbatch)
    echo "stage W arm $ARM: $W"
    for TRACK in qa nlq; do
        E=$(submit "$W" ARM=$ARM CKPT="$CKPT_HOME/W_$ARM" TRACK=$TRACK -- scripts/ego4d_eval.sbatch)
        echo "  eval $TRACK: $E"
    done
    if [ "$ARM" = A ]; then
        Q=$(submit "$W" TASK=q2 ARM=A CKPT="$CKPT_HOME/W_A" K="${K:-2}" -- scripts/ego4d_calibrate_q2.sbatch)
        echo "  q2: $Q"
    fi
done
squeue -u "$USER"
