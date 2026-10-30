# Ablation plots
Generated 2026-10-30 03:29:51 UTC from 5 complete stream groups (s000_010, s010_020, s020_030, s030_040, s040_050): 867 questions, 50 validation streams, 18 variants, trained model W_A (evaluation only, no training).
Error bars: 95% bootstrap intervals (2000 resamples) that resample whole streams. Differences between variants are paired on the same questions. Colours follow the entity across figures: full model blue, text only orange, oracle evidence aqua, random evidence gray.

## Figures
- `01_change_vs_full.png`: Variant minus full model for Recall@M, contains-match and exact-match, with 95% bootstrap intervals over streams.
- `02_retrieval_vs_answer.png`: Each variant: Recall@M against contains-match accuracy, with a linear trend line.
- `03_evidence_budget.png`: Recall@M, contains-match and exact-match against the number of evidence tokens (0, 1, 2, 4 = full, 8).
- `04_random_full_oracle.png`: Recall@M and contains-match with random, learned (full) and oracle evidence.
- `05_components.png`: Ranking-score terms, graph parts and the open segment: absolute Recall@M, best-node IoU and contains-match, with 95% intervals.
- `06_by_delay.png`: Recall@M and contains-match by how long before the question the answer moment occurred.
- `07_by_template.png`: Contains-match accuracy for each question template under six variants.
- `08_answer_agreement.png`: Pairwise share of questions on which two variants produce the same (normalised) answer string.
- `09_hit_vs_miss.png`: Contains-match accuracy split by whether the answer moment was among the retrieved evidence (Recall@M above 0), for four variants.
- `10_per_stream_difference.png`: Per-stream contains-match of the full model minus the text-only baseline, sorted.
- `11_stability_across_stream_groups.png`: Three key contrasts in contains-match, computed separately for each group of 10 streams, with 95% intervals.
- `12_open_vs_committed.png`: Recall@M and contains-match by where the answer moment lies in memory, with and without the open segment.

## Tables
- `results_table.csv`: every metric with 95% intervals, per variant
- `deltas_vs_full.csv`: change vs the full model with 95% intervals
- `report.pdf`: all figures in one file
