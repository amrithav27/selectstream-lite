# Results index

The reported numbers come from **`outputs/full_v3/`** (Ada, 4 × 2080 Ti, Qwen2.5-VL-7B fp16):
`collected_results.md` has every run's mean with a 95% CI and every comparison with a paired CI and a
permutation p-value (Holm-corrected within each family). Regenerate with
`python scripts/collect_results.py outputs/full_v3 outputs/full_v3/collected_results.json`.

| Folder | Who / where | What | Status |
|---|---|---|---|
| `outputs/full_v3/eval_*` | Ada, 45099/45872 | arms A–D, untrained, B with Stage-R writer; QA 243 / NLQ 300 val streams | reported |
| `outputs/full_v3/read_*` | Ada, 45873 + 75224 | 17 read-side ablations + 6 controls (text-only, zero / noise evidence, no_mark_read, seed_topk 8 ± coarse), first 150 val QA streams (2,338 q.), paired with `read_full` | reported |
| `outputs/full_v3/q2_*` | Ada, 45872 | Q2 composite questions, k = 2, 3, arms A, B | reported (CIs from per-item rows) |
| `outputs/full_v3/eval_qa_A_N*` | Ada, 45872 | memory budget N = 16/64/128, 9 long streams | case study; all-stream N = 64, 128 in `budget_qa_A_N*` (76739) |
| `outputs/full_v3/ablate_*` | Ada, 43319 | offline write / read grids (CPU) | reported; write grid rerun in `ablate_write_rerun/` reproduces it exactly (RF-19) |
| `results/colab_read_ablations/` | Colab A100 | the same 17 read ablations + text-only on the first 50 val streams | subset of the 150-stream run on other hardware; superseded by `read_*` |
| `results/colab_evidence_controls/` | Colab A100 (Dedeepya) | oracle / wrong / content-blanked evidence, text-only; blank and g-vector frames | supporting; real-frame versions in `outputs/full_v3/read_*` |
| `results/colab_p2_p3_p4/` | Colab A100 (Keerthana) | P2 metric breakdowns, P3 oracle frames, P4 seed 43 for W_A / W_B | supporting; Ada seeds 43, 44 in `outputs/full_v3/eval_*_seed*` |
| `selectstream_rf_runs_full/` (repo root) | Colab (Dedeepya) | reference + coarse_k4, topk8, no_mark_read, zero / noise evidence on all 243 val QA streams; `paired_vs_ref_243.json` | replicates the Ada 150-stream controls (report, replication table) |
| `selectstream_rf_runs/` (repo root) | Colab (Dedeepya) | same controls, first 50 streams | superseded by the 243-stream run |

Red-flag runs (`scripts/ego4d_redflags.sbatch`, jobs 75224 and its rerun 76739) land in `outputs/full_v3/` as `read_<control>`,
`W_<arm>_seed<s>`, `eval_<track>_<arm>_seed<s>`, `budget_qa_A_N<n>`, `ablate_write_rerun/`.

Colab results ran on different hardware (A100, single GPU) and are not pooled with the Ada runs.
