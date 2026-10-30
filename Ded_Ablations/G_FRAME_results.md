# Evidence controls with the cached g vector as the query frame (no blank frame)

Run: 2026-10-29/30, Colab A100. QA track, first 150 val QA streams (2,338 questions), arm A (`W_A`), one run each.
New flag `--g_frame` (`main/cli/eval_ego4d.py`): every visual token of the asked frame is replaced by that frame's cached
g_t (the mean of its projected tokens), so no pixels are needed. Token count and positions are those of the real frame.
The memory itself was built from the cached g vectors in every run, as before. Command = earlier runs with `--g_frame`
instead of `--blank_frame`, plus `--read_ablation` / `--evidence_k` as below.

## Results (95% bootstrap CI over videos)

| Run | contains | exact | recall@M |
|---|---|---|---|
| full (normal read) | 0.2070 [0.182, 0.231] | 0.1133 | 0.503 |
| oracle evidence | 0.2070 [0.182, 0.231] | 0.1133 | 0.934 |
| wrong evidence (non-overlapping nodes) | 0.2074 [0.183, 0.232] | 0.1138 | 0.022 |
| right nodes, content blanked | 0.2074 [0.183, 0.232] | 0.1133 | 0.934 |
| no evidence tokens (M = 0) | 0.1420 [0.122, 0.162] | 0.0013 | 0 |

Paired differences on contains: oracle - full = 0.0000; wrong - oracle = +0.0004 [0.0000, 0.0014];
nocontent - oracle = +0.0004 [0.0000, 0.0013]; M0 - full = -0.065 [-0.086, -0.046].
Predicted strings differ from the oracle run in only 15 (wrong) and 9 (nocontent) of 2,338 questions.

## Reading
- The earlier blank-frame result holds with the g-vector frame: the answers do not depend on which memory nodes are read
  or on what they contain. Retrieval that hits the answer window 93% of the time (oracle), 50% (normal read) and 2% (wrong)
  all score contains 0.207.
- Removing the evidence tokens altogether does change the output: contains drops 0.065, and exact match collapses
  (0.113 -> 0.001), because without evidence the model answers in full sentences instead of short phrases (as in P1).
  So the evidence tokens act as an answer-format cue; their content is not used.
- The g-vector frame adds a little over a blank frame: no-evidence 0.142 (g frame) vs 0.127 (P1, blank); with evidence
  0.207 (g frame) vs 0.200 (blank). The team's real-frame full model is 0.225.

## Caveats
- The g-frame is a stand-in for pixels (the model never trained on mean-of-token frames), so these numbers are not equal to a
  real-frame run. The team's real-frame `bound_oracle` run would be the check.
- One checkpoint, one run per setting, 150 of 243 streams. Contains is a loose substring metric.
- Arm A only. The A/B/C runs (`ORACLE_EVIDENCE_results.md`) were blank-frame.

## Raw files
`Ded_Ablations/g_frame/out/g_{full,oracle,wrong,nocontent,M0}/` (predictions.jsonl, summary.json), logs in `Ded_Ablations/g_frame/`.
Zip in Downloads: `g_results.zip`.
