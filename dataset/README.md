# dataset/

Ego4D NLQ and QaEgo4D annotations, cleaned into one format on the official NLQ splits.
No video files live here; clips are on Google Drive (`MyDrive/ego4d_nlq/clips/<clip_uid>.mp4`).

**Use `clean/` for everything downstream.** The other folders are raw sources, kept unchanged.

```
dataset/
├── annotations/            raw Ego4D NLQ (v2): nlq_train.json, nlq_val.json, nlq_test_unannotated.json
├── metadata/ego4d.json     raw Ego4D per-video / per-clip metadata (full corpus)
├── qaego4d/                raw QaEgo4D download
│   ├── download_qaego4d.py     fetches the annotations (no videos) from HF Ryougetsu/qaego4d, pinned revision
│   └── {train,val,test}_qa.json  rows as published, plus parsed moment_{start,end}_sec
├── clean/                  ← the final data
│   ├── build_clean.py          builds everything below from the raw folders
│   ├── verify_clean.py         checks every invariant listed under "Guarantees"; exits non-zero on failure
│   ├── build_videos.py         merges each video's clips into one stream; clip/video/stream time (below)
│   ├── clips.json
│   ├── videos_{train,val}.json one row per video, clips merged: the final form for the pipeline
│   ├── nlq_{train,val}.json
│   └── qaego4d_{train,val}.json
└── viz/                    browser visualiser over 100 videos (reads clean/ only; see viz/README.md)
```

Regenerate:

```bash
python3 -I qaego4d/download_qaego4d.py --out_dir qaego4d   # only if re-downloading
python3 -I clean/build_clean.py
python3 -I clean/build_videos.py
python3 -I clean/verify_clean.py   # also checks videos_*.json, so run it after build_videos
python3 -I viz/build_manifest.py
```

## Final format

### `clean/clips.json` — one row per clip

| field | meaning |
|---|---|
| `clip_uid` | Ego4D clip id. Unique. Also the Drive file name. |
| `video_uid` | parent Ego4D video. One video has one or more clips. |
| `split` | `train`, `val` or `test`: the official NLQ split. Every clip of a video has the same split. |
| `duration_sec` | clip length, from NLQ `clip_end_sec - clip_start_sec` (every clip starts at 0). |
| `participant_id` | `fb_participant_id` from `ego4d.json` (camera wearer); use it for participant-disjoint slices. |
| `drive_file_id` | Google Drive file id if the clip is on our Drive, else `null`. |

### `clean/{nlq,qaego4d}_{train,val}.json` — one row per query

All four files have exactly these fields, in this order:

| field | NLQ rows | QaEgo4D rows |
|---|---|---|
| `query_id` | `nlq:<annotation_uid>_<i>` (`i` = index in NLQ `language_queries`) | `qa:<sample_id>` |
| `source` | `"nlq"` | `"qaego4d"` |
| `split` | NLQ split | NLQ split of the clip (QaEgo4D train → `train`, QaEgo4D val **and test** → `val`) |
| `video_uid`, `clip_uid` | as in `clips.json` | as in `clips.json` (QaEgo4D's `video_id` is actually a clip_uid) |
| `query` | NLQ question text | QaEgo4D question text |
| `answer` | NLQ's optional free-text answer with the `"Answer (Optional):"` prefix removed; `null` for ~92% | QaEgo4D answer, always present |
| `clip_start_sec`, `clip_end_sec` | evidence window, seconds from clip start | NLQ's window if linked, otherwise QaEgo4D's own |
| `template`, `slot_x`, `verb_x`, `slot_y`, `verb_y` | from NLQ (may be null) | copied from the linked NLQ query; `null` if unlinked |
| `nlq_query_id` | its own `query_id` | the NLQ query it is the same question as, or `null` |

Rows with the same `nlq_query_id` are the same question. Join on it to put an NLQ template next to a QaEgo4D answer.

### Counts

| | clips | videos | NLQ queries | QaEgo4D QAs (linked to NLQ) |
|---|---|---|---|---|
| train | 1,271 (all on Drive) | 933 | 13,846 | 10,665 (9,714) |
| val | 415 (98 on Drive) | 303 | 4,552 | 3,751 (3,396) |
| test | 333 (none on Drive) | 258 | — (unannotated) | — |

Clip lengths are 210–1,200 s (median 480 s; 74 clips are over 500 s).

### `clean/videos_{train,val}.json` — one row per video, clips merged (the final form for the pipeline)

Built by `build_videos.py` from the files above, checked by `verify_clean.py` (2026-10-27, run on Ada).
One row per `video_uid` with at least one query: all its clips merged into one stream, every query on
three time axes.

| axis | seconds from | use |
|---|---|---|
| clip (`clip_start_sec`, `clip_end_sec`) | start of the query's own clip | clip-level comparison with published NLQ numbers |
| video (`video_start_sec`, `video_end_sec`) | start of the source video = clip `video_start_sec` + clip time | real elapsed time, gaps included |
| **stream** (`stream_start_sec`, `stream_end_sec`) | start of the merged stream | what the model has watched: gaps take no time, an overlapped moment counts once. Use for ask times, delays and Δt |

**Video row**

| field | meaning |
|---|---|
| `video_uid`, `split`, `participant_id` | as in `clips.json` |
| `video_duration_sec` | full source video (`ego4d.json`); only the annotated clips exist as mp4 |
| `stream_duration_sec` | length of the merged stream = sum of the played parts of the clips |
| `num_clips`, `num_nlq`, `num_qaego4d` | for ordering / bucketing streams by size |
| `clips` | video order. `video_start_sec`/`video_end_sec`; `play_from_sec` = clip time from which this clip plays (before it, an earlier clip already showed the same moments; 0 if no overlap); `plays` = false if the clip lies inside an earlier one; `stream_start_sec`/`stream_end_sec` of its played part; `overlap_prev_sec`, `gap_before_sec` |
| `gaps` | `video_start_sec`, `video_end_sec`, `gap_sec`, `stream_sec` (where on the stream the jump happens), `before_clip_uid` |
| `nlq`, `qaego4d` | the clean query rows unchanged, plus the video and stream times and `shown_by_clip_uid` (the clip whose frames show the window; differs from `clip_uid` when the window is in an overlap). Sorted by `stream_end_sec`. Two lists: the tracks are never merged for evaluation |

| | videos | 1 / 2 / ≥ 3 clips | clips overlapping the previous | clips inside an earlier one | gaps | windows shown by an earlier clip | queries | stream median / max |
|---|---|---|---|---|---|---|---|---|
| train | 924 | 706 / 137 / 81 | 77 (median 30 s, max 720 s) | 2 | 270 (median 330 s, max 4,689 s) | 226 | 24,511 | 480 / 3,330 s |
| val | 300 | 229 / 43 / 28 | 27 (all ~30 s) | 0 | 88 (median 240 s, max 2,310 s) | 91 | 8,303 | 480 / 3,060 s |

No question appears twice across overlapping clips (same text, answer and video window), so nothing is
dropped by merging. Example, Cooking `0e6fb738`: 5 clips spanning video 1:00–58:35 of a 3,521-s video, stream 2,335 s;
gaps of 90 s and 1,030 s; clips 4 and 5 start 30 s inside the previous clip and play from 0:30.

**How each merge issue is handled**

| Issue | In the data | Left to the pipeline |
|---|---|---|
| Gaps (unannotated video between clips) | recorded with length and stream position; stream time skips them | whether to force a cut at a gap; measure Δt and delays in stream seconds so a gap is not a jump |
| Overlaps (~30 s between consecutive clips) | earliest clip shows the moment (`play_from_sec`); each window says which clip shows it | decode clip k from `play_from_sec` (the loader already drops the repeated frames, same rule) |
| Fewer, longer items (924 streams vs 1,271 clips) | `stream_duration_sec`, query counts per video | ordering / bucketing; per-query backward keeps memory flat |
| Not comparable to clip-level NLQ results | clip-relative times kept on every query | report a clip-level number separately if wanted; our protocol (online, generated ask times) differs anyway |

**Checks** (`verify_clean.py`, all pass): every clean query appears in exactly one video, field values
identical to the clean row; clips complete and in video order; video time = offset + clip time; played
parts tile the stream with no hole; each window has the same length on the stream as on the video
(never spans a gap) and lies in the played part of `shown_by_clip_uid` at the stated stream time;
queries sorted by `stream_end_sec`.

## Cleaning, rule by rule

Each rule says what was found in the raw data and what `build_clean.py` does about it.

1. **Splits follow NLQ exactly.** QaEgo4D's own split is replaced by the NLQ split of the clip.
   QaEgo4D train clips are all NLQ train clips; QaEgo4D val and test clips are all NLQ val clips, so
   QaEgo4D test merges into `val`. Consequence: numbers on our `val` are not comparable to published
   QaEgo4D test numbers.
2. **One id per object.** A clip is identified by `clip_uid`, a video by `video_uid`, a query by
   `query_id`. QaEgo4D's `video_id` is renamed `clip_uid`, because it is an NLQ clip_uid, not a video.
   NLQ's video-level `split` field, `ego4d.json` `split_em` and the NLQ file a video sits in all agree.
3. **One time base: seconds from clip start.** NLQ also gives video-relative times, but they are
   inconsistent: for 14 train/val clips NLQ's clip-level `video_end_sec` is 2 frames longer than
   `ego4d.json` and the HF manifest say, and for 115 queries `video_*_sec ≠ clip_*_sec + clip offset`
   (by up to 0.53 s). Clip-relative times agree with `ego4d.json` and QaEgo4D, so only they are kept.
4. **QaEgo4D timecodes are `M:SS.FF` with FF = frame number at 30 fps**, not hundredths of a second
   (`0:43.19` = 43.633 s). Evidence: read as decimals, they sit 0–0.66 s before NLQ, always earlier; FF
   never exceeds 29; with FF/30 98% of matched windows agree with NLQ to within 0.05 s. One row has
   `4:20:01`; read as `4:20.01`, since a 4-hour reading is impossible for a ≤1,200 s clip.
5. **Windows clamped to `[0, clip duration]`.** NLQ: 339 windows end past the clip (264 by ≤0.05 s,
   73 by ≤1 s, 2 by 3.9 and 4.6 s). QaEgo4D: 156 (120 by ≤0.05 s, 36 by more). Three QaEgo4D point
   moments start 0.003–0.03 s past a 480 s clip end (within one frame) and are clamped onto the end.
6. **NLQ queries with no question text dropped:** 4.
7. **QaEgo4D rows with a redaction placeholder dropped:** 32 (`<object>`, `<she>`, `<objbct>` in the
   question or answer; e.g. answer `"<she>"`). These include the only two QaEgo4D `sample_id`s that
   were not unique, so every surviving `qa:` id is unique.
8. **Exact duplicates dropped** (same clip, normalised question, answer and window): 1 NLQ, 2 QaEgo4D.
   All come from one NLQ annotation (`842b098e…`) entered twice.
9. **QaEgo4D ↔ NLQ linking.** A QaEgo4D row is linked when it is in the same clip, its question text
   equals the NLQ text after lower-casing and stripping punctuation, and both window ends agree within
   0.05 s. 13,110 rows link (2 had two candidates; the closest window was taken). A linked row takes
   NLQ's window (QaEgo4D's is a frame-floored copy), template and slots. The 1,306 unlinked rows keep
   their own window and have null template.
10. **QaEgo4D point moments are kept.** 1,250 QaEgo4D rows (909 train, 341 val), all of them unlinked,
   have `clip_start_sec == clip_end_sec`: the raw start and end timecodes are identical, i.e. a single
   frame. NLQ has no zero-length windows. 988 of them match no NLQ question in their clip; 262 match an
   NLQ question by text, and for those the point lies inside the NLQ window 174 times, on its start or
   end 23 times, outside it 65 times. The points are spread across the clip (only 11 within 1 s of the
   end), and the questions are ordinary evidence questions, so they are read as single-instant
   evidence, not as the time a question was asked. That reading is an inference from the data, not
   documented by QaEgo4D. Downstream: a zero-length window has `|g| = 0`, so any metric dividing by the
   window length (T-Overlap, Eq. 57) must widen it to one frame period, as `_temporal_overlap` in
   `main/trainer/method_sft.py` already does for point-like intervals.

## Guarantees (checked by `verify_clean.py`)

- every `clip_uid` unique; every video in exactly one split
- every query row: known clip, `video_uid` and `split` equal to the clip's, file name matches split
  and source, `0 ≤ clip_start_sec ≤ clip_end_sec ≤ duration_sec` (equality allowed: point moments),
  non-empty question
- every `query_id` unique across all four files; no exact duplicates within a file
- QaEgo4D rows: non-empty answer, no `<…>` placeholder
- every `nlq_query_id` on a QaEgo4D row points to an existing NLQ row with the same clip, split and window

## Not verified / known caveats

- **Clip mp4 lengths on Drive are not checked** (no video on this laptop). `duration_sec` comes from
  the annotations; confirm with `ffprobe` on Ada before trusting window clamping near clip ends.
- **Participants overlap across splits**: 124 of the 156 val participants also appear in train. This is
  the official NLQ split, not something cleaning changed. Anything learned per wearer or per scene
  (e.g. a learned importance scorer) should also be reported on the participant-disjoint part of val.
- **NLQ and QaEgo4D are mostly the same questions** (13,110 linked pairs). Training on one and
  evaluating on the other within a split is not an independent test.
- QaEgo4D's `answers.json` (66 extra train/val ids with no question attached) is not used.
