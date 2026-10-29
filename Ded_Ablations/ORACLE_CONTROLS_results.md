# Oracle-evidence controls: wrong evidence and content-blanked evidence (arm A, blank frame)

Run: 2026-10-30, Colab A100. QA track, first 150 val QA streams (2,338 questions), arm A (`W_A`), blank frame, one run each.
Same command as `ORACLE_EVIDENCE_results.md`, with `--read_ablation` set to:
- `oracle_wrong`: evidence = the tightest memory nodes that do NOT overlap the asked window (same stream).
- `oracle_nocontent`: the right (oracle) nodes are selected, but their content is replaced by the memory mean before
  the evidence projection. Which nodes were read is unchanged; what they say is gone.
Reference: `oracle` on arm A (`Ded_Ablations/oracle_blank/`). New flags are in `main/model/dmg_gar.py` and `main/cli/eval_ego4d.py`.

## Results (95% bootstrap CI over videos)

| Evidence given to the reader | contains | exact | recall@M |
|---|---|---|---|
| oracle (right nodes, right content), reference | 0.2006 [0.177, 0.224] | 0.1116 | 0.934 |
| wrong nodes (do not overlap the window) | 0.2002 [0.176, 0.224] | 0.1112 | 0.022 |
| right nodes, content blanked | 0.2006 [0.177, 0.224] | 0.1116 | 0.934 |
| no evidence at all (P1 text-only, same blank frame) | 0.127 | 0.004 | n/a |

Paired differences on contains: wrong - oracle = -0.0004 [-0.0013, 0.0000]; nocontent - oracle = 0.0000.
The predicted strings differ in only 11 (wrong) and 10 (nocontent) of 2,338 questions.

## Reading
- With a blank frame, the reader's answers do not depend on what the evidence says. Evidence from the wrong part of the
  video (recall 0.02) and evidence with its content removed both give the same accuracy as the right evidence.
- So the jump from 0.127 (text-only) to 0.200 is not due to evidence content. It comes from the reader being given
  evidence tokens at all (P1 gave none), or from some other difference between the two runs; this run does not separate those.
- This corrects `ORACLE_EVIDENCE_results.md`, which read the 0.201 vs 0.127 gap as "accuracy depends on retrieved evidence
  being right". The new controls contradict that.
- Taken together with the arm-A/B/C result: with a blank frame, neither the write rule nor the retrieved content changes
  answer accuracy in this reader.

## Caveats
- Blank frame only (no mp4s). The question text is then the only other input, so the reader can lean on its language prior.
  Whether the same holds with a real frame is untested here (the team's `bound_oracle` run used a real frame).
- One checkpoint, one run per setting, 150 of 243 streams. Contains is a loose metric (substring match).
- The nocontent aggregate numbers equal the reference's because the same nodes were selected; the 10 differing
  predictions net out. Compare the predicted strings, not only the aggregates.
- A test of "any evidence tokens vs none" (for example, evidence of zeros or random noise) would settle the first bullet
  under Reading; not run.

## Raw files
`Ded_Ablations/oracle_controls/` (per-mode `predictions.jsonl`, `summary.json`, logs). Zip in Downloads: `ctrl_results.zip`.
