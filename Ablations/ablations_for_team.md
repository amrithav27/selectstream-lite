# selectstream-lite on Ego4D: pending ablations and how to run them

For teammates running on Google Colab. Checkpoints and results: Hugging Face model repo
`amritha27/selectstream-lite-ego4d`. Feature cache: dataset repo `amritha27/selectstream-lite-ego4d-features`. Last updated 2026-10-29.

## What the project tests (one paragraph)

A streaming video-QA model keeps a small memory (16 nodes) while watching an egocentric video and
answers questions about earlier moments. The backbone (Qwen2.5-VL-7B) is frozen; we train a writer
(what to store), a reader (what to retrieve) and an injector (how retrieved memory reaches the
answer). **Q1:** does a learned importance score beat the paper's "surprise" signal for deciding what
to write? **Q2:** does splitting a composite question into sub-questions retrieve better?

## Already done or running: don't redo these

| Group | What | Status |
|---|---|---|
| Main Q1 | Write arms A (surprise), B (learned scorer), C (oracle labels), D (fixed-length segments): Stage W + eval on QA and NLQ | A, B done; C, D finishing 2026-10-29 |
| Offline write ablations | memory size, segment length, scorer losses/inputs/label tolerance, boundary detector, which write decisions use the score | done |
| Trained vs untrained | same streams, untrained modules | done |
| B with Stage-R writer | what writer retraining buys | finishing 2026-10-29 |
| Q2 | single vs decomposed vs unmatched retrieval, 2–3 sub-questions, arms A and B | finishing 2026-10-29 |
| Memory size N = 64 / 128 at eval | 14 longest val streams | finishing 2026-10-29 |
| Read side | ranking-score terms, graph hops / GAR / re-rank, random and oracle evidence, evidence count M = 0/1/2/8, no open segment, blank frame | queued, done ~2026-10-30 morning |

Results so far, and the eval outputs, are in the Hub repo (`README.md`, `eval/`).

**Main open problem:** the answers barely use the memory. Arms A and B retrieve different evidence for
3,741 of 3,751 QA questions but give the same answer on 99.4% of them, and accuracy is the same whether
retrieval hits the answer moment or not (0.215 vs 0.226). Most pending items below test or fix this.

---

## Pending ablations

Ordered by priority. "Code" says whether it can run today or needs code first.

### P1. Text-only baseline (no frame, no memory)
- **Question:** how much of the QA accuracy comes from the question wording alone?
- **Code:** runs today (existing flags).
- **Run:** eval with a blank frame and no evidence (M = 0), checkpoint `W_A`, same 150 QA streams as the
  read ablations (see "Running an eval" below):
  `--blank_frame --evidence_k 0 --max_clips 150`
- **Cost:** ~25 min on 4 GPUs.
- **Compare with:** `read_full`, `read_M0`, `read_blank_frame`. If text-only ≈ full on contains-match,
  the answers are a language prior.

### P2. Standard metrics and breakdowns from existing predictions (CPU only)
- **Question:** comparable numbers and the breakdowns the plan promises.
- **Code:** needs a short script over `eval/*/predictions.jsonl` (each row has `window`,
  `evidence_spans`, `prediction`, `answer`, `template`, `linked`, `participant_disjoint`,
  `evidence_location`, `delay_bucket`).
- **What to compute:**
  - NLQ-style R@1 / R@5 at IoU ≥ 0.3 and 0.5, from the ranked `evidence_spans` vs `window`.
  - Lenient answer scoring (token F1, normalised match) next to exact / contains.
  - Every metric by participant-disjoint vs seen, linked vs unlinked, template, evidence location
    (open / committed), delay bucket.
  - Paired bootstrap over videos (resample `stream_id`) for every A-vs-X difference.
- **Cost:** minutes on a laptop; no GPU.

### P3. Oracle frames and uniform frames (plain Qwen, no memory module)
- **Question:** can the frozen backbone answer at all when shown the right frames? And how does the
  memory compare with the usual "sample frames from the past" baseline?
- **Code:** needs an eval flag (not written). Give Qwen2.5-VL the cached PNG frames as images:
  - *oracle frames:* up to 8 frames from the answer window;
  - *uniform frames:* 8 frames evenly spaced from stream start to t_q.
  Frames come from `FrameCache.image(clip_uid, k)` (`main/data/frame_cache.py`).
- **Cost:** ~1 h coding, ~30 min GPU each on 150 QA streams.
- **Reading:** oracle frames high + oracle evidence low → the memory injection is the bottleneck.
  Both low → the metric / answer format is the problem.

### P4. Two more training seeds (AB-R1)
- **Question:** are the A-vs-B differences stable across training runs, not just across videos?
- **Code:** runs today, but there is **no `--seed` flag**: the seed is `training.seed` in the config.
  Copy `configs/ego4d_qwen25vl7b.yaml` to e.g. `configs/ego4d_seed43.yaml` with `seed: 43`, and use it
  **for Stage W only**. **Evaluate with the original config**: the seed also fixes the question ask
  times, and every eval must ask at the same frames.
- **Run:** Stage W for arms A and B from `R_A`, seeds 43 and 44; eval each on QA and NLQ.
- **Cost:** ~8 h (4 Stage W × 50 min + 8 evals × 35 min).

### P5. Longer writer training (AB-W2), conditional
- Run only if "B with Stage-R writer" ≈ "B with Stage-W writer" (result due 2026-10-29).
- Stage W with `--max_steps 200` for A and B, then evals. ~3.5 h per arm plus evals.

### P6. Smaller backbone: Qwen2.5-VL-3B
- **Question:** does a different backbone also ignore the memory? Is the method size-dependent?
- **Code:** two small fixes first: hidden size 3584 is hard-coded in
  `main/model/importance_scorer.py` (`load_scorer` default) and `main/cli/replay_write.py`
  (`ImportanceScorer(3584, ...)`); read it from the model config instead. New config with
  `model_name_or_path: Qwen/Qwen2.5-VL-3B-Instruct`.
- **Run:** the whole pipeline (new feature cache with `main.cli.cache_features`, scorer, Stage R,
  calibration, Stage W, evals). Run it on 4 streams first.
- **Cost:** ~1.5 days on 4 × 2080 Ti (unmeasured on Colab). The new cache needs only `g/` + `meta/`
  (~7 GB at hidden size 2048); frames come from the clips via the fallback.

### P7. Memory-forcing training (a fix for the main problem, not an ablation)
- **Idea:** during Stage R, blank the current frame with probability p (e.g. 0.5) so the answer must
  come from memory.
- **Code:** needs a training flag (not written); the eval side already has `--blank_frame`.
- **Cost:** one Stage R (~7 h) + Stage W + evals.

### P8. Q2 items the plan lists but the code doesn't have
- A model-written split (the frozen Qwen splits the composite question, instead of the oracle split).
- Native two-hop NLQ questions ("Where is X before / after Y?").
- **Code:** needs adding to `main/cli/eval_q2.py`. See `docs/plan.md` §7.

---

## Setup on Google Colab

You already have the code (GitHub), the Ego4D NLQ clips and the cleaned annotations (`dataset/clean/`).
From the Hub you need only the **checkpoints** and the **feature cache**.

**GPU.** The frozen Qwen2.5-VL-7B is ~15.5 GB in fp16. Use an **A100 (40 GB)** runtime; an L4 (24 GB)
should fit evaluation but is tight for training; a T4 (16 GB) won't fit. Use fp16, never bf16 on older
GPUs (no native bf16 below compute capability 8.0).

**Code version.** The checkpoints need the code commit that adds the retrieval changes (newer than
`f7aa8cd`) and the frame fallback below. Ask Amritha for the commit to check out.

**Packages.** We ran `torch` 2.7.1, `transformers` 5.5.4, `huggingface_hub` 1.11.0. `transformers`
5.5.4 refuses to import with `huggingface_hub` ≥ 2.0; keep it below 2.0.

**1. Download** (Colab disk: checkpoints ~10 GB, features 12 GB)

```python
from huggingface_hub import login, snapshot_download
login()  # your own token, read access
CK = snapshot_download("amritha27/selectstream-lite-ego4d", local_dir="/content/ck",
                       allow_patterns=["checkpoints/full_v3/*", "runs/full_v3/scorer/*",
                                       "runs/full_v3/calibration.json", "config/*"])
FEAT = snapshot_download("amritha27/selectstream-lite-ego4d-features", repo_type="dataset",
                         local_dir="/content/feature_cache")
```

Drop items from `allow_patterns` you don't need (e.g. only `checkpoints/full_v3/W_A/*` for read-side
ablations). `runs/` has every result and log if you want to compare.

**2. Feature cache without frames.** The Hub cache has `g/` (per-frame features) and `meta/` but not the
183 GB of PNG frames. With `--feature_cache` *and* `--video_dir` pointing at your clips
(`<clip_uid>.mp4`), the code decodes only the frame each question is asked at, from your mp4, with the
same decoder call that built the cache. Pixel-identical to the cached frames on Ada [[CHECKED: see
Amritha's note]]; a different OpenCV build on Colab may differ slightly.

**3. Single-GPU config.** Our config shards the backbone over 4 GPUs. Copy
`configs/ego4d_qwen25vl7b.yaml` and change one line:

```yaml
  max_memory: {0: "38GiB"}   # was {0: "1GiB", 1: "10GiB", 2: "10GiB", 3: "10GiB"}
```

`vismem.trainable_device: "cuda:0"` stays as is.

**4. Run an eval** (e.g. P1, text-only):

```bash
python -m main.cli.eval_ego4d --config configs/ego4d_colab.yaml --arm A \
    --ckpt /content/ck/checkpoints/full_v3/W_A --track qa \
    --video_dir /content/clips --feature_cache /content/feature_cache \
    --scorer /content/ck/runs/full_v3/scorer/scorer.pt \
    --calibration /content/ck/runs/full_v3/calibration.json \
    --out_dir /content/out/read_text_only --max_clips 150 \
    --blank_frame --evidence_k 0
```

- `--arm` must match the checkpoint (`W_B` → `--arm B`).
- Other switches: `--read_ablation {no_prior,no_cos,no_bias,cos_only,prior_only,no_hops,no_gar,coarse_only,random,oracle}`,
  `--evidence_k M`, `--no_open`, `--blank_frame`.
- Stage W (P4, P5) from `checkpoints/full_v3/R_A`: `python -m main.cli.train_ego4d --stage W --arm A
  --init_from /content/ck/checkpoints/full_v3/R_A --max_steps 50 --calibration ... --scorer ...
  --feature_cache ... --video_dir ... --output_dir ...`
- Colab sessions end; Stage W writes `latest/` every 10 streams and resumes from it, so put
  `--output_dir` on persistent storage.
- Times above were measured on 4 × RTX 2080 Ti; an A100 should be faster but this is not measured.

---

## Giving access to the Hub repo (for Amritha)

The repo is private under a personal account. Two ways to share it (from my knowledge of the Hub;
check the current Hugging Face docs before relying on it):

1. **Organisation (recommended):** create a free organisation on huggingface.co (New → Organization),
   invite teammates as members with the *read* role, then move the repo there (repo Settings → Rename
   or transfer → `<org>/selectstream-lite-ego4d`). Teammates then download with their own login.
2. **Gated repo:** make the repo public but gated with manual approval (repo Settings → Access
   requests). Teammates click "request access" and you approve each one; nobody else gets files.

Don't share your own access token: it gives write access to everything on your account.
