# selectstream-lite

What should a streaming video memory keep? A rebuild of SelectStream [1] (arXiv 2606.16353) with a frozen
Qwen2.5-VL-7B, on Ego4D NLQ + QaEgo4D, run strictly online. Two questions:

- **Q1 (write side):** does an importance score *learned* from training questions decide what to keep better
  than [1]'s surprise signal, at the same memory budget and segment count?
- **Q2 (read side):** does splitting a composite question into sub-questions beat one retrieval at the same
  evidence budget?

Report: [`docs/report.md`](docs/report.md) (LaTeX source in `docs/report_overleaf/`). Every number in it is in
`outputs/full_v3/collected_results.md`.

## Layout

```
streamlite/
  model/        backbone wrapper, LVM memory (write path), DMG/GAR reader (read path), importance scorer (Q1)
  cli/          cache_features, train_scorer, calibrate_write, train_ego4d (Stage R / W), eval_ego4d, eval_q2,
                ablate_read / ablate_write (offline), replay_write, plot_ego4d
  data/         Ego4D stream dataset, frame / feature cache
  trainer/      retrieval loss and metrics shared by training and evaluation (method_sft.py)
configs/        ego4d_qwen25vl7b.yaml (the reported runs); ego4d_colab*.yaml (single A100)
dataset/        README.md + build scripts for the cleaned annotations (the JSONs are not redistributed)
scripts/        Ada (Slurm) jobs, collect_results.py (every comparison with CI + p), diagnostics/
outputs/full_v3/  the reported run: summary.json + predictions.jsonl per evaluation, figures/,
                  collected_results.{json,md}, ckpt_verify.json
results/        teammates' Colab runs (smaller subsets, A100); see results/README.md
docs/           the report (report.md, report_overleaf/)
```

## Reproducing

**Hardware.** All reported runs: 4 × RTX 2080 Ti (11 GB) on one node, backbone in fp16 (no native bf16 on
these GPUs), trainable modules in fp32. GPU times are in the report (§ Experiment runs); the main chain
(cache, Stage R, Stage W A–D, evaluations, Q2) is roughly 30 GPU-hours of node time.

**1. Environment.** Python 3.11.

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**2. Data.** Ego4D NLQ v2 annotations and clips need an Ego4D licence (<https://ego4d-data.org>). Put the raw
annotations in `dataset/annotations/` and the Ego4D metadata in `dataset/metadata/`, then build the cleaned
splits (details and guarantees in `dataset/README.md`):

```bash
cd dataset
python3 -I qaego4d/download_qaego4d.py --out_dir qaego4d
python3 -I clean/build_clean.py && python3 -I clean/build_videos.py && python3 -I clean/verify_clean.py
```

The 1,686 annotated clips (`<clip_uid>.mp4`) go in `$VIDEO_DIR` (set in `scripts/ego4d_env.sh`).

**3. Pipeline.** 

## Running on Ada

The reported run (`full_v3`) is three Slurm jobs on **one** 4 × RTX 2080 Ti node. All three must use the same
node (`-w`): the frame-feature cache (194 GB) and the checkpoints live in that node's `/ssd_scratch`, which other
nodes cannot see. Plan for ~400 GB free scratch (clips 148 GB, cache 194 GB, model 16 GB, checkpoints).

**Set up once, on the login node** (`ssh <user>@ada.iiit.ac.in`; light work only there):

```bash
git clone https://github.com/amrithav27/selectstream-lite.git ~/selectstream-lite && cd ~/selectstream-lite
conda create -n venv python=3.11 -y                 # scripts use ~/miniforge3/envs/venv/bin/python (PY in scripts/ego4d_env.sh)
~/miniforge3/envs/venv/bin/pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
~/miniforge3/envs/venv/bin/pip install -r requirements.txt
~/miniforge3/envs/venv/bin/python -c "from huggingface_hub import login; login()"   # token for the model download
myallocation                                        # expect account research, QoS medium, partition u22
mkdir -p ~/bin && cp /usr/bin/rclone ~/bin/         # compute nodes have no rclone; jobs use ~/bin/rclone
rclone config                                       # once: add a Google Drive remote named "gdrive"
```

Do not run `hf auth login` or upgrade `huggingface_hub`: version 2.x breaks `transformers` 5.5.4, and every job
checks this at start. Build `dataset/clean/` as in step 2 (CPU, login node is fine). The clips are pulled from
Google Drive with `rclone` (remote `gdrive:`, path `ego4d_nlq/clips`; both set in `scripts/ego4d_env.sh`).

**Run** (pick a free 2080 Ti node, e.g. from `sinfo -N -p u22 -o "%N %t %f"`, and use it for every job):

```bash
NODE=gnode047
export NODES=$NODE                                  # this node holds the data (default: gnode061 gnode044)
sbatch -w $NODE scripts/ego4d_stage_all.sbatch      # 0. copy all clips from Drive into $NODE's scratch (CPU only)

# 1. Main chain (~30 h): feature cache -> scorer -> Stage R -> calibrate -> Stage W per arm (A-D) -> evals -> Q2
P=$(RUN=full_v3 sbatch --parsable -w $NODE scripts/ego4d_pipeline.sbatch)

# 2. Read-side ablations on 150 val QA videos + offline write-side grid (~8 h)
A=$(RUN=full_v3 sbatch --parsable -w $NODE --dependency=afterok:$P scripts/ego4d_ablate_read.sbatch)

# 3. Follow-up controls (~8 h): checkpoint hash check, evidence controls, Stage W seeds 43/44 for A and B,
#    memory size N = 64/128 on all QA videos, write-grid rerun, then collect_results.py
RUN=full_v3 sbatch -w $NODE --dependency=afterok:$A scripts/ego4d_controls.sbatch

squeue -u $USER                                     # watch the queue
tail -f logs/ss_pipe-$P.out                         # logs: logs/ss_pipe-*, ss_ablate-*, ss_controls-*
```

**Figures and statistics** (CPU; regenerate after any job finishes):

```bash
srun -A research -p u22 -c 2 --mem=6G -t 01:00:00 \
  ~/miniforge3/envs/venv/bin/python -m streamlite.cli.plot_ego4d --run_dir outputs/full_v3        # outputs/full_v3/figures/
srun -A research -p u22 -c 2 --mem=6G -t 01:00:00 \
  ~/miniforge3/envs/venv/bin/python scripts/collect_results.py outputs/full_v3 outputs/full_v3/collected_results.json
```

**Notes.**
- Every stage writes `outputs/full_v3/done.<stage>`; resubmitting the same job to the same node skips finished
  stages, so a job that hits the wall limit or fails is simply resubmitted.
- Small results (summaries, predictions, logs, figures) land in `outputs/full_v3/` in the repo; weights stay in
  scratch and are mirrored to `~/selectstream_ckpt/full_v3/`.
- Node scratch is wiped after 7 days without access; jobs touch their files, but a long pause means re-staging.
- The GPUs have no native bf16, so the backbone runs in fp16 (set in `configs/ego4d_qwen25vl7b.yaml`). Every job
  logs which GPUs it got.
- Seeds and settings of the reported run are the script defaults (`R_EPOCHS=2`, `W_STEPS=50`, seeds 42 / 43 / 44,
  `READ_STREAMS=150`); no extra variables are needed.

**Without retraining.** Checkpoints of the reported run are on the Hugging Face Hub:
<https://huggingface.co/lma-project/selectstream-lite-ego4d> (`checkpoints/full_v3/{R_A,W_A,W_B,W_C}/weights.pt`;
W_D and the seed 43/44 writers are not uploaded). The Q1 scorer is `outputs/full_v3/scorer/scorer.pt`. Evaluate
one arm with

```bash
python -m streamlite.cli.eval_ego4d --arm A --ckpt <dir with weights.pt> --track qa --video_dir <clips> \
    --scorer outputs/full_v3/scorer/scorer.pt --calibration outputs/full_v3/calibration.json --out_dir <out>
```

The frame-feature cache (194 GB) is not hosted; `streamlite.cli.cache_features` rebuilds it (7.2 h on 4 GPUs).
Without `--feature_cache` the evaluation computes features on the fly (slower, same features up to cosine 0.99998).

## What is trained

The backbone (Qwen2.5-VL-7B) is frozen. Trained, 436.1M parameters in fp32: segment encoder, write gate + MLP,
question encoder, re-ranker and evidence projection, with [1]'s loss (answer + 0.1 × retrieval + 0.05 × sparsity).
Stage R trains reader and writer with arm A (surprise); Stage W freezes the reader and retrains the writer per arm
(A surprise, B learned scorer, C oracle labels, D fixed length) on the same streams and steps. The Q1 scorer is a
separate 0.06M-parameter network (`streamlite/cli/train_scorer.py`). Three retrieval changes are ours, not [1]'s:
a learned score temperature, centred segment features, and a question–node similarity prior.
