"""Point merged_{train,val}.json at the clips in YOUR Google Drive.

Run it in Colab (the Drive API call uses the notebook's Google login; the account must be able to open the folder):

    from google.colab import auth; auth.authenticate_user()
    !python retarget_merged_drive_ids.py --folder_id <YOUR_CLIPS_FOLDER_ID>

It walks the folder (and sub-folders, following shortcuts) and matches a clip to a file when the clip_uid appears
in the file name (e.g. "<clip_uid>.mp4"). In every clip entry of the merged files, drive_file_id becomes the id
in your folder; clips with no file get null and are listed at the end, with the videos they leave incomplete.
Inputs are never overwritten: merged_train.json -> merged_train_mydrive.json (same for val), or pass --suffix.
Needs update_drive_ids.py in the same folder (it provides the folder walk). Run without -I here.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from update_drive_ids import UUID, list_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_id", required=True, help="Drive folder id that holds <clip_uid>.mp4 files")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--suffix", default="_mydrive")
    args = ap.parse_args()

    from googleapiclient.discovery import build
    service = build("drive", "v3")

    by_uid, seen = collections.defaultdict(list), 0
    for fid, name, _ in list_folder(service, args.folder_id):
        seen += 1
        for uid in set(UUID.findall(name.lower())):
            by_uid[uid].append(fid)
    print(f"files seen in folder: {seen}")

    for split in args.splits:
        src = f"{HERE}/merged_{split}.json"
        out = f"{HERE}/merged_{split}{args.suffix}.json"
        merged = json.load(open(src))
        matched, missing, partial = 0, [], []
        for v in merged:
            lost = 0
            for c in v["clips"]:
                hits = by_uid.get(c["clip_uid"])
                c["drive_file_id"] = hits[0] if hits else None
                if hits:
                    matched += 1
                else:
                    lost += 1
                    missing.append(c["clip_uid"])
            if lost:
                partial.append(v["video_uid"])
        with open(out, "w") as f:
            json.dump(merged, f, indent=1, ensure_ascii=False)
        print(f"{split}: {matched} clips matched, {len(missing)} with no file, "
              f"{len(partial)}/{len(merged)} videos incomplete -> {out}")
        if missing:
            print("  first clips with no file:", missing[:5])


if __name__ == "__main__":
    main()
