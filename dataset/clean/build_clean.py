"""Build one clean, single-format view of Ego4D NLQ + QaEgo4D, on the official NLQ splits.

Inputs (read-only): ../annotations/nlq_{train,val,test_unannotated}.json, ../metadata/ego4d.json,
../qaego4d/{train,val,test}_qa.json, "../viz/ego4d_clip_ids .csv".

Outputs, all in this folder:
  clips.json                 one row per clip: clip_uid, video_uid, split, duration_sec, participant_id,
                             drive_file_id (null if the clip is not on our Drive), video_start_sec
                             (clip offset in its video, ego4d.json)
  nlq_{train,val}.json       one row per NLQ query
  qaego4d_{train,val}.json   one row per QaEgo4D QA; QaEgo4D train -> train, QaEgo4D val AND test -> val
Query rows in all four files share one schema (QUERY_FIELDS). Times are clip-relative seconds only.

Cleaning rules:
  - NLQ queries with no query text are dropped (4 rows).
  - QaEgo4D rows with a redaction placeholder such as <object> or <she> are dropped.
  - Windows are clamped to [0, clip duration]; annotations overshoot clip ends by up to ~4.6 s.
  - Exact duplicates are dropped (DEDUP_KEY: same clip, normalised text, window, and answer); they all
    come from one NLQ annotation entered twice and its QaEgo4D answers.
  - QaEgo4D moments are M:SS.FF with FF = frame at 30 fps, floored, so they sit up to 1 frame before NLQ.
    A QaEgo4D row is linked to an NLQ query (nlq_query_id) when it is in the same clip, its normalised
    question text equals the NLQ text, and both window ends agree within LINK_TOL; linked rows take the
    NLQ window, template and slots. Unlinked rows keep their own window and have null template/slots.
  - NLQ's video-relative times are not carried over: for 14 train/val clips NLQ's clip-level
    video_end_sec disagrees with ego4d.json by 2 frames, and 115 query-level video times disagree
    with clip time + clip offset. Clip-relative times are consistent with ego4d.json and QaEgo4D.

    python3 -I build_clean.py
"""
import collections
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
LINK_TOL = 0.05
PLACEHOLDER = re.compile(r"<[^<>]+>")
NLQ_ANSWER_PREFIX = "Answer (Optional):"
QUERY_FIELDS = [
    "query_id", "source", "split", "video_uid", "clip_uid", "query", "answer",
    "clip_start_sec", "clip_end_sec", "template", "slot_x", "verb_x", "slot_y", "verb_y", "nlq_query_id",
]


def norm_text(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).split()


def dedup_key(r):
    return (r["clip_uid"], tuple(norm_text(r["query"])), tuple(norm_text(r["answer"] or "")),
            r["clip_start_sec"], r["clip_end_sec"])


def dedup(rows, stats, label):
    seen, out = set(), []
    for r in rows:
        k = dedup_key(r)
        if k in seen:
            stats[f"{label} dropped: exact duplicate"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out


def clamp(s, e, dur):
    return max(0.0, min(s, dur)), max(0.0, min(e, dur))


def main():
    stats = collections.Counter()
    meta = json.load(open(f"{DS}/metadata/ego4d.json"))
    participant = {v["video_uid"]: v.get("fb_participant_id") for v in meta["videos"]}
    # Clip offset inside its source video, from ego4d.json (the source durations are checked against);
    # the loader uses it to replay a video's clips as one stream.
    clip_offset = {c["clip_uid"]: c["video_start_sec"] for c in meta["clips"]}
    with open(f"{DS}/viz/ego4d_clip_ids .csv") as f:
        drive = {r[0].strip(): r[1].strip() for r in csv.reader(f) if r}

    clips, nlq_rows = {}, {"train": [], "val": []}
    for split, fn in [("train", "nlq_train"), ("val", "nlq_val"), ("test", "nlq_test_unannotated")]:
        for v in json.load(open(f"{DS}/annotations/{fn}.json"))["videos"]:
            for c in v["clips"]:
                cu, dur = c["clip_uid"], c["clip_end_sec"] - c["clip_start_sec"]
                clips[cu] = {
                    "clip_uid": cu, "video_uid": v["video_uid"], "split": split, "duration_sec": round(dur, 3),
                    "participant_id": participant.get(v["video_uid"]), "drive_file_id": drive.get(cu),
                    "video_start_sec": clip_offset.get(cu, c["video_start_sec"]),
                }
                stats["clip offset: ego4d.json vs NLQ differ > 0.1 s"] += abs(clip_offset.get(cu, c["video_start_sec"]) - c["video_start_sec"]) > 0.1
                stats["clip offset: missing in ego4d.json (NLQ value used)"] += cu not in clip_offset
                if split == "test":
                    continue
                for a in c["annotations"]:
                    for i, q in enumerate(a.get("language_queries", [])):
                        if not q.get("query"):
                            stats["nlq dropped: no query text"] += 1
                            continue
                        s, e = clamp(q["clip_start_sec"], q["clip_end_sec"], dur)
                        stats["nlq window clamped to clip"] += (s, e) != (q["clip_start_sec"], q["clip_end_sec"])
                        ans = (q.get("answer") or "").removeprefix(NLQ_ANSWER_PREFIX).strip() or None
                        qid = f"nlq:{a['annotation_uid']}_{i}"
                        nlq_rows[split].append({
                            "query_id": qid, "source": "nlq", "split": split, "video_uid": v["video_uid"],
                            "clip_uid": cu, "query": q["query"].strip(), "answer": ans,
                            "clip_start_sec": round(s, 3), "clip_end_sec": round(e, 3),
                            "template": q.get("template"), "slot_x": q.get("slot_x"), "verb_x": q.get("verb_x"),
                            "slot_y": q.get("slot_y"), "verb_y": q.get("verb_y"), "nlq_query_id": qid,
                        })

    by_clip = collections.defaultdict(list)
    for split in nlq_rows:
        nlq_rows[split] = dedup(nlq_rows[split], stats, "nlq")
        for r in nlq_rows[split]:
            by_clip[r["clip_uid"]].append(r)

    qa_rows, seen_ids = {"train": [], "val": []}, collections.Counter()
    for qa_split in ["train", "val", "test"]:
        for r in json.load(open(f"{DS}/qaego4d/{qa_split}_qa.json")):
            if PLACEHOLDER.search(r["question"]) or PLACEHOLDER.search(r["answer"]):
                stats["qaego4d dropped: redaction placeholder"] += 1
                continue
            c = clips[r["video_id"]]
            split = c["split"]
            assert split == ("train" if qa_split == "train" else "val"), (r["sample_id"], qa_split, split)
            s, e = clamp(r["moment_start_sec"], r["moment_end_sec"], c["duration_sec"])
            stats["qaego4d window clamped to clip"] += (s, e) != (r["moment_start_sec"], r["moment_end_sec"])
            cands = [n for n in by_clip[c["clip_uid"]] if norm_text(n["query"]) == norm_text(r["question"])
                     and abs(n["clip_start_sec"] - s) <= LINK_TOL and abs(n["clip_end_sec"] - e) <= LINK_TOL]
            link = min(cands, key=lambda n: abs(n["clip_start_sec"] - s) + abs(n["clip_end_sec"] - e)) if cands else None
            stats["qaego4d linked to NLQ"] += link is not None
            stats["qaego4d linked, >1 candidate (closest window taken)"] += len(cands) > 1
            seen_ids[r["sample_id"]] += 1
            qid = f"qa:{r['sample_id']}" + (f"#{seen_ids[r['sample_id']]}" if seen_ids[r["sample_id"]] > 1 else "")
            src = link or {}
            qa_rows[split].append({
                "query_id": qid, "source": "qaego4d", "split": split, "video_uid": c["video_uid"],
                "clip_uid": c["clip_uid"], "query": r["question"].strip(), "answer": r["answer"].strip(),
                "clip_start_sec": src.get("clip_start_sec", round(s, 3)), "clip_end_sec": src.get("clip_end_sec", round(e, 3)),
                "template": src.get("template"), "slot_x": src.get("slot_x"), "verb_x": src.get("verb_x"),
                "slot_y": src.get("slot_y"), "verb_y": src.get("verb_y"),
                "nlq_query_id": link["query_id"] if link else None,
            })
    stats["qaego4d duplicate sample_id renamed with #n"] = sum(n - 1 for n in seen_ids.values() if n > 1)
    for split in qa_rows:
        qa_rows[split] = dedup(qa_rows[split], stats, "qaego4d")

    def dump(name, obj):
        with open(f"{HERE}/{name}", "w") as f:
            json.dump(obj, f, indent=1, ensure_ascii=False)

    dump("clips.json", sorted(clips.values(), key=lambda c: (c["split"], c["video_uid"], c["clip_uid"])))
    for split in ["train", "val"]:
        dump(f"nlq_{split}.json", nlq_rows[split])
        dump(f"qaego4d_{split}.json", qa_rows[split])
        print(f"{split}: nlq {len(nlq_rows[split])} queries, qaego4d {len(qa_rows[split])} QAs")
    print(f"clips: {collections.Counter(c['split'] for c in clips.values())}, on Drive: {sum(c['drive_file_id'] is not None for c in clips.values())}")
    for k, n in stats.items():
        print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
