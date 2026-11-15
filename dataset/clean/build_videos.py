"""Video-level view of the clean data: one row per video_uid, all its annotated clips merged into one
stream, every query on three time axes. This is the final form the pipeline should read.

Inputs (read-only, this folder): clips.json, nlq_{train,val}.json, qaego4d_{train,val}.json;
../metadata/ego4d.json for the full video length.
Outputs, this folder: videos_{train,val}.json (field list in ../README.md).

Time axes (seconds):
  clip    from the start of the query's own clip (as in the clean files; kept for clip-level comparison)
  video   from the start of the source video = clip video_start_sec + clip time
  stream  from the start of the merged stream: what a viewer of the stream has watched so far. Gaps
          (video no clip covers) take no stream time; a moment shown by two overlapping clips is shown
          once, by the earlier clip, and takes stream time once.

How the merge handles each issue:
  overlap   clip k plays only from play_from_sec (clip time): before that, an earlier clip already
            showed the same video moments. A clip inside an earlier one plays nothing (plays=false).
            A query's window is mapped through the clip that actually shows it (shown_by_clip_uid).
  gap       recorded in `gaps` with its length and the stream time where the stream jumps over it, so
            the pipeline can cut there and measure elapsed time (e.g. Δt) in stream seconds.
  size      stream_duration_sec and query counts per video, for ordering / bucketing streams.

Checks (assertions): every query lands in exactly one video; clip time + offset = video time; a window
lies inside its clip; on the stream axis a window has the same length as on the video axis (it never
spans a gap); played pieces tile [0, stream_duration_sec] without holes.
Same frame rule as main/data/ego4d_qa.py (overlapping frames shown once, earliest clip wins).

    python3 -I build_videos.py
"""
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.dirname(HERE)
EPS = 1e-3


def r3(x):
    return round(float(x), 3)


def merge_clips(cs):
    """cs sorted by video_start_sec -> (clip rows, gaps, played pieces [(video_a, video_b, stream_a, clip_uid)])."""
    rows, gaps, pieces = [], [], []
    seen_end, stream = None, 0.0
    for c in cs:
        a, b = float(c["video_start_sec"]), float(c["video_start_sec"]) + float(c["duration_sec"])
        gap = a - seen_end if seen_end is not None and a > seen_end else 0.0
        overlap = max(0.0, min(seen_end, b) - a) if seen_end is not None else 0.0
        if gap > 0:
            gaps.append({"video_start_sec": r3(seen_end), "video_end_sec": r3(a), "gap_sec": r3(gap),
                         "stream_sec": r3(stream), "before_clip_uid": c["clip_uid"]})
        play_a = a if seen_end is None else max(a, seen_end)
        plays = b > play_a
        if plays:
            pieces.append((play_a, b, stream, c["clip_uid"]))
        rows.append({
            "clip_uid": c["clip_uid"], "drive_file_id": c["drive_file_id"], "duration_sec": c["duration_sec"],
            "video_start_sec": r3(a), "video_end_sec": r3(b),
            "play_from_sec": r3(min(play_a, b) - a), "plays": plays,
            "stream_start_sec": r3(stream), "stream_end_sec": r3(stream + max(0.0, b - play_a)),
            "overlap_prev_sec": r3(overlap), "gap_before_sec": r3(gap),
        })
        stream += max(0.0, b - play_a)
        seen_end = b if seen_end is None else max(seen_end, b)
    return rows, gaps, pieces, stream


def to_stream(t, pieces):
    """Video time -> (stream time, clip showing it). Pieces are contiguous on the stream axis."""
    for a, b, s0, cu in pieces:
        if a - EPS <= t <= b + EPS:
            return s0 + min(max(t - a, 0.0), b - a), cu
    raise AssertionError(f"video time {t} is not shown by any clip")


def main():
    clips = json.load(open(f"{HERE}/clips.json"))
    clip_by_uid = {c["clip_uid"]: c for c in clips}
    clips_by_video = collections.defaultdict(list)
    for c in clips:
        clips_by_video[c["video_uid"]].append(c)
    full_len = {v["video_uid"]: v.get("duration_sec") for v in json.load(open(f"{DS}/metadata/ego4d.json"))["videos"]}

    for split in ["train", "val"]:
        rows_by_video = collections.defaultdict(lambda: {"nlq": [], "qaego4d": []})
        n_in = 0
        for src in ["nlq", "qaego4d"]:
            for r in json.load(open(f"{HERE}/{src}_{split}.json")):
                c = clip_by_uid[r["clip_uid"]]
                assert c["video_uid"] == r["video_uid"] and c["split"] == split, r["query_id"]
                rows_by_video[r["video_uid"]][src].append(r)
                n_in += 1

        out, n_out, stats = [], 0, collections.Counter()
        for vid in sorted(rows_by_video):
            cs = sorted(clips_by_video[vid], key=lambda c: c["video_start_sec"])
            assert {c["split"] for c in cs} == {split}, vid
            clip_rows, gaps, pieces, stream_len = merge_clips(cs)
            # played pieces tile the stream axis
            assert abs(pieces[0][2]) < EPS and all(abs(p[2] + (p[1] - p[0]) - q[2]) < EPS for p, q in zip(pieces, pieces[1:]))
            qs = {}
            for src, rows in rows_by_video[vid].items():
                qs[src] = []
                for r in rows:
                    c = clip_by_uid[r["clip_uid"]]
                    off = float(c["video_start_sec"])
                    vs, ve = off + r["clip_start_sec"], off + r["clip_end_sec"]
                    assert off - EPS <= vs <= ve <= off + c["duration_sec"] + EPS, r["query_id"]
                    ss, shown_by = to_stream(vs, pieces)
                    se, _ = to_stream(ve, pieces)
                    assert abs((se - ss) - (ve - vs)) < 2 * EPS, (r["query_id"], vs, ve, ss, se)  # never spans a gap
                    stats["windows shown by an earlier, overlapping clip"] += shown_by != r["clip_uid"]
                    qs[src].append({**r, "video_start_sec": r3(vs), "video_end_sec": r3(ve),
                                    "stream_start_sec": r3(ss), "stream_end_sec": r3(se), "shown_by_clip_uid": shown_by})
                qs[src].sort(key=lambda q: (q["stream_end_sec"], q["query_id"]))
                n_out += len(qs[src])
            stats["clips overlapping an earlier clip"] += sum(c["overlap_prev_sec"] > 0 for c in clip_rows)
            stats["clips inside an earlier clip (play nothing)"] += sum(not c["plays"] for c in clip_rows)
            stats["gaps"] += len(gaps)
            out.append({
                "video_uid": vid, "split": split, "participant_id": cs[0]["participant_id"],
                "video_duration_sec": full_len.get(vid), "stream_duration_sec": r3(stream_len),
                "num_clips": len(clip_rows), "num_nlq": len(qs["nlq"]), "num_qaego4d": len(qs["qaego4d"]),
                "clips": clip_rows, "gaps": gaps, "nlq": qs["nlq"], "qaego4d": qs["qaego4d"],
            })
        assert n_out == n_in, (n_out, n_in)  # every query in exactly one video
        with open(f"{HERE}/videos_{split}.json", "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        lens = sorted(v["stream_duration_sec"] for v in out)
        ncl = collections.Counter(min(v["num_clips"], 3) for v in out)
        print(f"{split}: {len(out)} videos, {n_out} queries; clips/video 1: {ncl[1]}, 2: {ncl[2]}, >=3: {ncl[3]}; "
              f"stream_duration_sec median {lens[len(lens) // 2]:.0f}, max {lens[-1]:.0f}")
        for k, n in stats.items():
            print(f"  {k}: {n}")


if __name__ == "__main__":
    main()
