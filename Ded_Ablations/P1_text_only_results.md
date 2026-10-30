# P1: text-only baseline (no frame, no memory)

Run: 2026-10-29, Colab A100, checkpoint `W_A` (`amritha27/selectstream-lite-ego4d`, `checkpoints/full_v3/W_A`), arm A, QA track,
first 150 val QA streams (2,338 questions), single run.

Command (Colab copy of the repo at commit `e6601a9`, config = the run's config with `max_memory: {0: "38GiB"}`):

    python -m main.cli.eval_ego4d --config configs/ego4d_colab.yaml --arm A --ckpt <W_A> --track qa \
        --video_dir /content/clips --feature_cache <hub feature cache> --scorer <scorer.pt> --calibration <calibration.json> \
        --out_dir /content/out/read_text_only --max_clips 150 --blank_frame --evidence_k 0

One local patch in the Colab copy only: with `--blank_frame` the frame size is read from the cache meta (`sizes[0]`)
instead of decoding the mp4, because no mp4s were available. The model sees a black frame of the same size either way.

## Result (paired by question against the stored `runs/full_v3/eval_qa_A/predictions.jsonl`)

| Metric | Full model | Text-only | Diff, 95% bootstrap CI over questions |
|---|---|---|---|
| contains | 0.225 | 0.127 | -0.098 [-0.115, -0.083] |
| exact | 0.119 | 0.004 | -0.115 [-0.129, -0.103] |

Text-only `summary.json` overall: contains 0.1266, exact 0.0038 (recall_at_m 0 and best_iou are not meaningful with no evidence).

## Reading
- The question wording alone gets contains 0.127, about 56% of the full model's 0.225: a large language prior.
- The frame and memory together add about 0.10 contains (CI excludes 0).
- Exact match is a format artefact: text-only answers in full sentences ("The trolley was on the floor before you ..."),
  the trained model gives short phrases ("on the floor"). Use contains.
- P1 removes frame and memory together. The memory's own share needs `read_M0` (frame, no memory) and
  `read_blank_frame` (memory, no frame) from the queued job.

## Caveats
- Single run, single checkpoint. The bootstrap resamples questions, not videos (questions within a video are correlated,
  so the true CI is wider).
- Colab used transformers 5.20.2 (the team used 5.5.4). The "full" column is the team's stored predictions, not a re-run here.
- The run was done twice (the first run's raw files were lost in a Colab reset). The second run reproduced the first exactly:
  contains 0.1266, exact 0.0038, paired diff -0.098 [-0.115, -0.083]. The saved raw files are from the second run.

## Raw files
`Ded_Ablations/p1_text_only/out/read_text_only/` in this repo folder: `summary.json` (all breakdowns), `predictions.jsonl`
(2,338 rows), `p1.log`. Also as a zip in Downloads: `p1_text_only_results.zip`. Not copied to Drive (the Drive mount could
not be approved through the Colab connection); drop the folder or zip into the Drive folder by hand.
