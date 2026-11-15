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

**3. Pipeline.** On a Slurm cluster (paths in `scripts/ego4d_env.sh`; adapt `SCRATCH`, `PY`, account and
partition to yours):

```bash
RUN=full_v3 sbatch -w <node> scripts/ego4d_pipeline.sbatch     # feature cache -> scorer -> Stage R -> calibrate
                                                                # -> Stage W (arms A-D) -> evals -> Q2
RUN=full_v3 sbatch -w <node> scripts/ego4d_ablate_read.sbatch  # read-side ablations, 150 val QA streams
RUN=full_v3 sbatch -w <node> scripts/ego4d_redflags.sbatch     # evidence controls, seeds 43/44, memory budget
python scripts/collect_results.py outputs/full_v3 outputs/full_v3/collected_results.json
```

Each stage writes `outputs/<RUN>/done.<stage>`; a resubmitted job skips finished stages. Run on the node that
holds the feature cache (`-w`).

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
