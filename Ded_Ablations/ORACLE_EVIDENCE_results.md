# Oracle evidence on arms A, B and C (blank frame)

Run: 2026-10-29, Colab A100. QA track, first 150 val QA streams (2,338 questions), one run per arm.
Checkpoints `W_A`, `W_B`, `W_C` from `amritha27/selectstream-lite-ego4d` (`checkpoints/full_v3/`).
Command per arm (Colab copy of the repo at `e6601a9`, run's config with `max_memory: {0: "38GiB"}`):

    python -m main.cli.eval_ego4d --config configs/ego4d_colab.yaml --arm {A,B,C} --ckpt W_{A,B,C} --track qa \
        --video_dir /content/clips --feature_cache <hub feature cache> --scorer <scorer.pt> --calibration <calibration.json> \
        --max_clips 150 --read_ablation oracle --blank_frame

Deviation from the planned run: `--blank_frame` was added because no mp4 clips were available (a real-frame eval decodes
each question's frame from the mp4). So the only visual information is the retrieved evidence. These numbers are NOT
directly comparable with the team's `bound_oracle` run (arm A, real frame).

## Results (95% bootstrap CI over videos)

| Arm | contains | exact | recall@M | best_iou |
|---|---|---|---|---|
| A (surprise) | 0.2006 [0.1775, 0.2247] | 0.1116 | 0.934 [0.892, 0.977] | 0.193 [0.170, 0.219] |
| B (learned score) | 0.2002 [0.1767, 0.2244] | 0.1112 | 0.943 [0.905, 0.982] | 0.170 [0.154, 0.189] |
| C (oracle write) | 0.2010 [0.1778, 0.2252] | 0.1116 | 0.910 [0.855, 0.965] | 0.199 [0.179, 0.220] |

Paired differences (same questions, bootstrap over videos):

| Contrast | contains | recall@M | best_iou |
|---|---|---|---|
| B - A | -0.0004 [-0.0014, 0.0000] | +0.009 [+0.001, +0.018] | -0.023 [-0.035, -0.011] |
| C - A | +0.0004 [-0.0009, +0.0020] | -0.024 [-0.042, -0.007] | +0.006 [-0.010, +0.021] |
| C - B | +0.0009 [-0.0008, +0.0027] | -0.033 [-0.055, -0.012] | +0.029 [+0.015, +0.042] |

## Reading
> CORRECTION (2026-10-30): the bullet below that says accuracy depends on retrieved evidence being right is contradicted
> by the controls in `ORACLE_CONTROLS_results.md`: wrong-window evidence and content-blanked evidence give the same
> contains (0.200). Read the 0.201 vs 0.127 gap as "tokens present vs absent", not "right evidence".

- With a perfect reader the answer accuracy is identical across write rules: contains 0.200-0.201 for A, B and C, with
  paired differences of at most 0.001. Even oracle-labelled writing (C) does not help the answers.
- The write rule still changes memory quality (best_iou: B lower than A by 0.023, CI excludes 0; same direction as
  the offline replay), but that does not reach the answers.
- Evidence matters when it is right: arm A with oracle evidence gets contains 0.201 versus 0.127 for text-only
  (P1: same blank frame, no evidence). The real frame and learned retrieval together score 0.225 (full model).
- Together with P1 this says: accuracy depends on retrieved evidence being correct, but not on which write rule
  built the memory. Q1 (learned importance vs surprise) is unlikely to show up in answer accuracy at this budget.

## Caveats
- One run per arm, one seed per checkpoint, blank frame, 150 of 243 val QA streams.
- recall@M under oracle read is high for all arms by construction; its small differences (0.91-0.94) are secondary.
- The text-only vs oracle comparison uses different runs on the same questions; I did not compute a paired CI for it.

## Raw files
`Ded_Ablations/oracle_blank/` in this repo folder (per-arm `predictions.jsonl`, `summary.json`, logs). Zip in Downloads:
`oracle_blank_results.zip`. Not on Drive (the Drive mount cannot be approved through the Colab connection).
