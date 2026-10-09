# selectstream-lite

SelectStream-lite on Ego4D NLQ. **[1]** = SelectStream, arXiv 2606.16353 v2.

An ablation of [1]'s memory-allocation policy, not a reproduction of it. Two questions:

- **Q1** — does a *learned* importance score decide what to keep better than [1]'s hand-set
  priority (surprise, access, recency)? Changes how the memory graph is built.
- **Q2** — does splitting a question into sub-questions and checking evidence for each beat a
  single retrieval step, **at an equal evidence budget**?

`docs/plan.md` is the source of truth for method, data and evaluation. `docs/selectstream_math.tex`
is the maths reference for [1]; `docs/selectstream_lite_changes.tex` is the companion that records
every deviation from it.

## Layout

Code lives under `source/`, one package per module of [1], so provenance is readable from the path.

```
source/
  config.yaml          every budget and default, each tagged with its source
  vlm/                 frozen backbone (Phi_v, p_theta) -- never updated
  saw/                 SAW, Surprise-driven Adaptive Windowing -- when to write
  lvm/                 LVM, Latent Visual Memory -- what to preserve   (+ Q1 scorer)
  gar/                 GAR, Graph Attention Reasoning -- how to read   (+ Q2 decomposition)
  train/               the two training loops
  data/                dataset plumbing
  eval/                metrics and the replay runner
annotations/ metadata/ raw Ego4D NLQ json          (gitignored)
docs/
```

Our two contributions sit inside the module they modify: Q1 changes how the graph is *built*
(`source/lvm/scorer.py`), Q2 changes how the query is *read* (`source/gar/decompose.py`). Each file's
docstring carries its `[1]` / `[1]-mod` / `ours` tag.

## Running

`source/` is a package. Run from the repo root so imports resolve:

```bash
python -m source.data.nlq                      # split stats + coverage, writes source/data/splits.json
python -m source.vlm.features <split> [limit] [rank] [world]
```

**Everything runs on Ada, never locally.** Use the full interpreter path
(`~/miniforge3/envs/venv/bin/python`) in batch jobs, fp16 rather than bf16, and log
`torch.cuda.get_device_name(0)` at job start. See `docs/plan.md` §7 for compute sizing.

## What is trained

The backbone VLM is frozen, exactly as in [1]. Three modules are trained: [1]'s `Enc_q` and GAR
(from a retrieval loss, since NLQ has no answer text and `L_ans` is not computable), plus the new Q1
scorer. [1]'s `SegEnc` and write gate are kept closed-form, and its calibration layer `W_e` is gone
because lite injects real keyframes rather than latent tokens. `docs/plan.md` §2 gives the reasoning
and the ordering constraint that Q1's fairness depends on.
