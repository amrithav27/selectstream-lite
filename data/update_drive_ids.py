"""Point clips.json at our own Drive copy of the Ego4D clips.

Run in Colab (the Drive API call needs the notebook's Google login):

    from google.colab import auth; auth.authenticate_user()
    !python update_drive_ids.py --folder_id <FOLDER_ID> --clips clips.json --out clips_mydrive.json

It walks the folder (and sub-folders, following folder shortcuts), reads each file's id and name, and matches
a file to a clip when the clip_uid appears in the file name (e.g. "<clip_uid>.mp4"). For matched clips,
drive_file_id is replaced by the id in the copy and the old value is kept as drive_file_id_orig. Clips with
no file in the folder get drive_file_id = null. The input file is never overwritten.

build_clean.py rebuilds clips.json from the original CSV, so write to a separate --out file.
"""
import argparse
import collections
import json
import re

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"


def list_folder(service, root_id):
    """Yield (id, name, parent_id) for every non-folder file under root_id."""
    queue, seen = [root_id], {root_id}
    while queue:
        parent = queue.pop()
        token = None
        while True:
            resp = service.files().list(
                q=f"'{parent}' in parents and trashed = false",
                fields="nextPageToken, files(id, name, mimeType, shortcutDetails(targetId, targetMimeType))",
                pageSize=1000, pageToken=token, supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            for f in resp["files"]:
                kind, fid = f["mimeType"], f["id"]
                if kind == SHORTCUT:
                    kind, fid = f["shortcutDetails"]["targetMimeType"], f["shortcutDetails"]["targetId"]
                if kind == FOLDER:
                    if fid not in seen:
                        seen.add(fid)
                        queue.append(fid)
                else:
                    yield f["id"], f["name"], f["mimeType"] == SHORTCUT
            token = resp.get("nextPageToken")
            if not token:
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder_id", required=True, help="Drive folder id of our copy of the clips")
    ap.add_argument("--clips", default="clips.json")
    ap.add_argument("--out", default="clips_mydrive.json")
    args = ap.parse_args()

    from googleapiclient.discovery import build
    service = build("drive", "v3")

    by_uid, stats = collections.defaultdict(list), collections.Counter()
    for fid, name, is_shortcut in list_folder(service, args.folder_id):
        stats["files seen"] += 1
        stats["file shortcuts seen (their id is the original file's)"] += is_shortcut
        for uid in set(UUID.findall(name.lower())):
            by_uid[uid].append((fid, name))

    clips = json.load(open(args.clips))
    for c in clips:
        hits = by_uid.get(c["clip_uid"], [])
        c["drive_file_id_orig"] = c.get("drive_file_id_orig", c.get("drive_file_id"))
        if not hits:
            c["drive_file_id"] = None
            stats["clips with no file in the copy"] += 1
            continue
        if len(hits) > 1:
            stats["clips with >1 matching file (first taken)"] += 1
        c["drive_file_id"] = hits[0][0]
        stats["clips matched"] += 1

    with open(args.out, "w") as f:
        json.dump(clips, f, indent=1, ensure_ascii=False)
    print(f"{len(clips)} clips -> {args.out}")
    for k, n in stats.items():
        print(f"  {k}: {n}")
    missing = [c["clip_uid"] for c in clips if c["drive_file_id"] is None][:5]
    if missing:
        print("first clips with no file:", missing)


if __name__ == "__main__":
    main()
