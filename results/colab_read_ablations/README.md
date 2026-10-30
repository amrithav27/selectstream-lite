# Read-side ablations for selectstream-lite on Ego4D (evaluation only, no training)

Generated 2026-10-30. Everything here evaluates the already-trained arm-A model (`W_A`, run `full_v3`); no weights were changed.

## What was run

Each run plays validation question-answering streams through `W_A` and scores the answers. One part of the reading step is changed or switched off at test time, and the score change shows how much that part contributes. 18 variants, each on the first 50 of the 243 validation streams (867 questions), run as 5 groups of 10 streams (90 runs in all).

| Variant | What it does |
|---|---|
| `full` | Nothing changed. The reference. |
| `text_only` | Black frame and no memory: the question wording alone. |
| `blank_frame` | The question is asked over a black frame; memory is kept. |
| `M0` | No memory: 0 evidence tokens (the model was trained with 4). |
| `M1`, `M2`, `M8` | 1, 2 or 8 evidence tokens instead of 4. |
| `bound_random` | Evidence picked at random (floor for retrieval). |
| `bound_oracle` | Evidence taken from the true answer moment (ceiling for retrieval). |
| `score_no_prior` | Ranking without the question-based prior term. |
| `score_no_cos` | Ranking without the question-to-node cosine term. |
| `score_no_bias` | Ranking without the small bias terms (surprise boost, span and merge penalties). |
| `score_cos_only` | Ranking by cosine similarity alone. |
| `score_prior_only` | Ranking by the prior alone. |
| `graph_no_hops` | No expansion to neighbouring memory nodes. |
| `graph_no_gar` | Graph reasoning layers switched off. |
| `graph_coarse_only` | First-pass ranking only, no refined re-ranking. |
| `no_open` | The segment still being written is hidden; only finished nodes can be retrieved. |

`text_only` is an extra baseline from the team doc (item P1); the other 17 are the read-side runs of the cluster job that did not run.

## Results (50 validation streams, 867 questions)

Point estimates; 95% intervals (bootstrap over streams) are in `results_table.csv`, and paired differences against `full` are in `deltas_vs_full.csv`. Recall@M = how often the answer moment is among the retrieved evidence. Contains-match = the reference answer appears in the prediction.

| Variant | Recall@M | contains | exact |
|---|---|---|---|
| full | 0.518 | 0.200 | 0.125 |
| text_only | 0.000 | 0.122 | 0.009 |
| blank_frame | 0.518 | 0.194 | 0.121 |
| M0 (no memory) | 0.000 | 0.128 | 0.000 |
| M1 | 0.142 | 0.197 | 0.125 |
| M2 | 0.281 | 0.207 | 0.126 |
| M8 | 0.835 | 0.203 | 0.128 |
| bound_random | 0.452 | 0.198 | 0.126 |
| bound_oracle | 0.925 | 0.198 | 0.126 |
| score_no_prior | 0.397 | 0.198 | 0.125 |
| score_no_cos | 0.491 | 0.200 | 0.125 |
| score_no_bias | 0.502 | 0.201 | 0.126 |
| score_cos_only | 0.513 | 0.198 | 0.125 |
| score_prior_only | 0.512 | 0.198 | 0.125 |
| graph_no_hops | 0.547 | 0.194 | 0.122 |
| graph_no_gar | 0.547 | 0.192 | 0.120 |
| graph_coarse_only | 0.547 | 0.198 | 0.126 |
| no_open | 0.489 | 0.197 | 0.125 |

What the numbers show (observations, not explanations):

- **Retrieval moves a lot; answers barely move.** Recall@M ranges from 0.14 (1 token) to 0.93 (oracle) across variants that pass evidence, yet contains-match stays between 0.19 and 0.21. Random evidence and oracle evidence give the same contains-match (0.198).
- **Only removing the evidence entirely changes the answers clearly.** `M0` is 0.072 below `full` in contains-match (95% interval -0.098 to -0.047) and `text_only` is 0.077 below (-0.103 to -0.050). Every other variant's interval for the contains-match change includes 0 or just touches it.
- **A black frame changes little.** `blank_frame` leaves retrieval unchanged and contains-match 0.006 lower (-0.028 to +0.019).
- **Retrieval components.** Removing the prior lowers Recall@M by 0.121 (-0.227 to +0.003); hiding the open segment lowers it by 0.029 (-0.052 to -0.007). Removing graph hops, graph reasoning or the refined re-ranking each raises Recall@M by 0.029 (0.002 to 0.057), with no clear change in answers.
- **Oracle evidence recalls the answer moment in 92.5% of questions**, not 100% (interval 0.85 to 1.00).

Limits: 50 of 243 validation streams, a few questions per stream, so the intervals are wide (for example contains-match for `full`: 0.166 to 0.237). Only 47 of the 867 questions come from participants not seen in training, too few for a separate seen/unseen comparison, so none is reported. The answer metric is string-based.

## Plots, tables and raw results

All stored in the Google Drive of **tiruveedulanikhilesh@gmail.com** (the account the notebook ran on), folder `selectstream_ablations/`:

- `plots/`: the figures below, `report.pdf` (all figures in one file), `INDEX.md`, `results_table.csv`, `deltas_vs_full.csv`.
- `out/s000_010/ ... s040_050/read_<variant>/`: `summary.json` and `predictions.jsonl` (every question's answer and retrieved evidence) for each of the 90 runs, plus one log per run.
- `ego4d_colab.yaml`: the single-GPU config used.

The figures, `report.pdf` and `INDEX.md` are copied into `plots/` in this folder. The full Drive folder, including the raw per-question outputs, is here: https://drive.google.com/drive/folders/19Cv1GFCpMaoFjSOjE5K_ITHoTKLY3x0s?usp=sharing

| Figure | Shows |
|---|---|
| `01_change_vs_full` | Each variant minus the full model for Recall@M, contains-match and exact-match, with 95% intervals. |
| `02_retrieval_vs_answer` | Recall@M against contains-match, one point per variant, with a trend line. |
| `03_evidence_budget` | The three metrics as the number of evidence tokens goes 0, 1, 2, 4, 8. |
| `04_random_full_oracle` | Retrieval and answer accuracy with random, learned and oracle evidence. |
| `05_components` | Ranking terms, graph parts and the open segment, in absolute values. |
| `06_by_delay` | Accuracy by how long before the question the answer moment happened. |
| `07_by_template` | Contains-match by question template, for six variants (heatmap). |
| `08_answer_agreement` | Share of questions on which two variants give the identical answer (18 by 18 heatmap). |
| `09_hit_vs_miss` | Contains-match when the answer moment was or was not retrieved. |
| `10_per_stream_difference` | Per-stream contains-match, full model minus text only. |
| `11_stability_across_stream_groups` | Three key contrasts computed separately for each group of 10 streams. |
| `12_open_vs_committed` | Recall@M and contains-match by whether the answer moment is in a finished or still-open segment. |

Each figure passed an automatic check for overlapping text; I could not look at the rendered images in the tool used, so give `report.pdf` a quick visual pass. Colours: full model blue, text only orange, oracle evidence aqua, random evidence gray.

## How to reproduce

The run was done in the Colab notebook `read_ablations.ipynb` (cells numbered Step 0 to Step 11). In order:

1. Use a GPU runtime with an A100 (40 GB).
2. Install `huggingface_hub<2.0` and `transformers==5.5.4` (Step 1).
3. Put the repo code (commit `e6601a9`, or current `main` with this PR), the cleaned annotations and the Drive ids of the 205 clips in `/content` (Step 2).
4. Download the 205 clips from the public Drive links (Step 3).
5. Log in to Hugging Face and download `W_A`, the scorer, `calibration.json` and the feature cache from `lma-project` (Step 4). This needs access to that organisation.
6. Write the single-GPU config (Step 5). `--start` is in the repo after this PR.
7. Run the smoke test, then the ablation loop (Steps 6 and 7; 7b runs it in a background thread). It takes about 9 hours of A100 time.
8. Make the plots and tables (Step 10), then finish and unmount Drive (Step 11).

## Notes on the code

The run used commit `e6601a9` of this repo (then under `main/`, now `streamlite/`) plus the `--start` flag, which this PR adds to `streamlite/cli/eval_ego4d.py`. `--start K --max_clips N` evaluates streams K to K+N-1; the 5 groups of 10 streams were `--start 0/10/20/30/40 --max_clips 10`. The Colab notebook is `read_ablations.ipynb` in this folder (a Drive file listing was removed from one cell's output). The figures are in `plots/` (copy of the Drive folder's `plots/`).
