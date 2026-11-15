"""Check every invariant of the clean files; exits non-zero on the first broken one.

    python3 -I verify_clean.py
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_clean import QUERY_FIELDS, dedup_key  # noqa: E402

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


clips = {c["clip_uid"]: c for c in json.load(open(f"{HERE}/clips.json"))}
check(len(clips) == len(json.load(open(f"{HERE}/clips.json"))), "clip_uid not unique in clips.json")
video_split = collections.defaultdict(set)
for c in clips.values():
    video_split[c["video_uid"]].add(c["split"])
    check(c["split"] in {"train", "val", "test"}, f"bad split {c}")
    check(c["duration_sec"] > 0, f"non-positive duration {c['clip_uid']}")
    check(c.get("video_start_sec") is not None and c["video_start_sec"] >= 0, f"bad video_start_sec {c['clip_uid']}")
for vu, s in video_split.items():
    check(len(s) == 1, f"video {vu} in splits {sorted(s)}")

rows = {}
for name in ["nlq_train", "nlq_val", "qaego4d_train", "qaego4d_val"]:
    split = name.split("_")[1]
    keys = set()
    for r in json.load(open(f"{HERE}/{name}.json")):
        qid = r["query_id"]
        check(dedup_key(r) not in keys, f"{name}: exact duplicate {qid}")
        keys.add(dedup_key(r))
        check(list(r) == QUERY_FIELDS, f"{name}: wrong fields {list(r)}")
        check(qid not in rows, f"duplicate query_id {qid}")
        rows[qid] = r
        c = clips.get(r["clip_uid"])
        check(c is not None, f"{qid}: clip not in clips.json")
        if c is None:
            continue
        check(r["split"] == split == c["split"], f"{qid}: split {r['split']} / file {split} / clip {c['split']}")
        check(r["video_uid"] == c["video_uid"], f"{qid}: video_uid differs from clips.json")
        check(0 <= r["clip_start_sec"] <= r["clip_end_sec"] <= c["duration_sec"], f"{qid}: window outside [0, duration]")
        check(bool(r["query"]), f"{qid}: empty query")
        check(r["source"] == name.split("_")[0], f"{qid}: source {r['source']} in {name}")
        if r["source"] == "qaego4d":
            check(bool(r["answer"]), f"{qid}: empty answer")
            check("<" not in r["query"] + r["answer"], f"{qid}: placeholder left")
        else:
            check(r["nlq_query_id"] == qid, f"{qid}: nlq row not self-linked")

for qid, r in rows.items():
    if r["source"] == "qaego4d" and r["nlq_query_id"]:
        n = rows.get(r["nlq_query_id"])
        check(n is not None and n["source"] == "nlq", f"{qid}: link target missing")
        if n:
            check((n["clip_uid"], n["split"], n["clip_start_sec"], n["clip_end_sec"]) ==
                  (r["clip_uid"], r["split"], r["clip_start_sec"], r["clip_end_sec"]), f"{qid}: link disagrees with target")

msg = f"{len(clips)} clips, {len(video_split)} videos, {len(rows)} query rows checked"

# Merged streams (build_videos.py): same queries, one stream per video, three consistent time axes.
TOL = 2e-3
in_videos = collections.Counter()
for split in ["train", "val"]:
    path = f"{HERE}/videos_{split}.json"
    if not os.path.exists(path):
        failures.append(f"{path} missing: run build_videos.py")
        continue
    for v in json.load(open(path)):
        vu, cs = v["video_uid"], v["clips"]
        check(v["split"] == split and video_split[vu] == {split}, f"{vu}: split")
        check([c["clip_uid"] for c in cs] == [c["clip_uid"] for c in sorted((clips[c["clip_uid"]] for c in cs), key=lambda c: c["video_start_sec"])]
              and {c["clip_uid"] for c in cs} == {u for u, c in clips.items() if c["video_uid"] == vu}, f"{vu}: clips not all / not in video order")
        played = 0.0
        for c in cs:
            check(abs(c["stream_start_sec"] - played) < TOL, f"{vu}/{c['clip_uid']}: stream_start not contiguous")
            played += (c["duration_sec"] - c["play_from_sec"]) if c["plays"] else 0.0
            check(abs(c["stream_end_sec"] - played) < TOL, f"{vu}/{c['clip_uid']}: stream_end")
            check(abs(c["video_start_sec"] - clips[c["clip_uid"]]["video_start_sec"]) < TOL, f"{vu}/{c['clip_uid']}: offset")
        check(abs(v["stream_duration_sec"] - played) < TOL, f"{vu}: stream_duration_sec")
        play_span = {c["clip_uid"]: c for c in cs}
        for src in ["nlq", "qaego4d"]:
            check(v[f"num_{src}"] == len(v[src]), f"{vu}: num_{src}")
            prev = -1.0
            for q in v[src]:
                qid = q["query_id"]
                in_videos[qid] += 1
                r = rows.get(qid)
                check(r is not None and all(q[k] == r[k] for k in QUERY_FIELDS), f"{qid}: differs from clean row")
                off = clips[q["clip_uid"]]["video_start_sec"]
                check(abs(q["video_start_sec"] - (off + q["clip_start_sec"])) < TOL and abs(q["video_end_sec"] - (off + q["clip_end_sec"])) < TOL,
                      f"{qid}: video time != offset + clip time")
                check(abs((q["stream_end_sec"] - q["stream_start_sec"]) - (q["video_end_sec"] - q["video_start_sec"])) < TOL, f"{qid}: stream length")
                s = play_span[q["shown_by_clip_uid"]]  # the clip that shows the window start must play it, at that stream time
                check(s["plays"] and s["video_start_sec"] + s["play_from_sec"] - TOL <= q["video_start_sec"] <= s["video_end_sec"] + TOL
                      and abs(s["stream_start_sec"] + (q["video_start_sec"] - s["video_start_sec"] - s["play_from_sec"]) - q["stream_start_sec"]) < TOL,
                      f"{qid}: stream time / shown_by clip")
                check(q["stream_end_sec"] >= prev, f"{qid}: not sorted by stream_end_sec")
                prev = q["stream_end_sec"]
check(set(in_videos) == set(rows) and all(n == 1 for n in in_videos.values()), "videos_*.json: a query missing or repeated")
msg += f"; videos_{{train,val}}.json: {len(in_videos)} queries"

print(msg)
if failures:
    print(f"{len(failures)} FAILURES, first 10:")
    for f in failures[:10]:
        print("  ", f)
    sys.exit(1)
print("All invariants hold.")
