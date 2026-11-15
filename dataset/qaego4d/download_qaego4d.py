"""Download the QaEgo4D annotations (no videos) from Ryougetsu/qaego4d on the Hugging Face Hub.

Writes {train,val,test}_qa.json: every QA row of that split, original fields unchanged, plus
moment_start_sec / moment_end_sec. The source moment strings are clip-relative M:SS.FF
timecodes where FF is a frame number at 30 fps, not hundredths of a second.

    python3 -I download_qaego4d.py --out_dir .
    python3 -I download_qaego4d.py --out_dir . --clips_csv "../viz/ego4d_clip_ids .csv"   # only clips we have
"""
import argparse
import csv
import json
import os
import re
import urllib.request

REPO = "Ryougetsu/qaego4d"
REVISION = "d7312012163b7cabe003e042c3a8ef6b92613ae8"  # pinned: the commit inspected on 2026-10-09
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/annotations.{{split}}.json"


def timecode_to_sec(tc: str) -> float:
    # One train row has "4:20:01": NLQ clips are at most 1200 s, so it is M:SS:FF with a typo, not hours.
    m = re.fullmatch(r"(\d+):(\d{2})[.:](\d{2})", tc.strip())
    if m is None:
        raise ValueError(f"Unrecognised timecode {tc!r}")
    minutes, seconds, frame = (int(g) for g in m.groups())
    return minutes * 60 + seconds + frame / 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--clips_csv", default=None, help="Keep only rows whose clip_uid is in column 0 of this CSV.")
    args = ap.parse_args()

    keep = None
    if args.clips_csv:
        with open(args.clips_csv) as f:
            keep = {row[0].strip() for row in csv.reader(f) if row}

    os.makedirs(args.out_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        with urllib.request.urlopen(URL.format(split=split), timeout=60) as resp:
            rows = json.load(resp)
        total = len(rows)
        if keep is not None:
            rows = [r for r in rows if r["video_id"] in keep]
        for r in rows:
            r["moment_start_sec"] = round(timecode_to_sec(r["moment_start"]), 3)
            r["moment_end_sec"] = round(timecode_to_sec(r["moment_end"]), 3)
        out = os.path.join(args.out_dir, f"{split}_qa.json")
        with open(out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"{split}: {len(rows)}/{total} QAs, {len({r['video_id'] for r in rows})} clips -> {out}")


if __name__ == "__main__":
    main()
