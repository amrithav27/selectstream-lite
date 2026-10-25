"""Local copies of the clips a sample reads, fetched from Google Drive the first time they are needed."""
from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path


def download_drive_file(video_path: Path, drive_id: str) -> None:
    """Fetch one Drive file to `video_path` (via a .part file, so an interrupted download is never used)."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = video_path.with_suffix(video_path.suffix + ".part")
    try:
        try:
            import gdown  # handles Drive's large-file confirmation page; preinstalled on Colab

            gdown.download(id=drive_id, output=str(tmp_path), quiet=True)
        except ImportError:
            request = urllib.request.Request(
                f"https://drive.google.com/uc?export=download&id={drive_id}", headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(tmp_path, "wb") as out:
                    shutil.copyfileobj(response, out)

        if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Empty download for Drive file {drive_id} (is it shared as 'anyone with the link'?)")
        # Reject an HTML error/confirmation page instead of caching it as a video.
        with open(tmp_path, "rb") as f:
            header = f.read(64)
        if b"ftyp" not in header:
            raise RuntimeError(
                f"Download for {drive_id} does not look like an MP4. "
                "Google Drive may have returned an error page."
            )
        os.replace(tmp_path, video_path)
        print(f"Cached video: {video_path}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_video_available(sample):
    """Make sure the video file(s) a sample reads exist locally, downloading from Drive only when first needed."""
    meta = sample.meta or {}
    merged = meta.get("merged_clips")
    if merged:
        # Only the clips the replay actually reads: those that start before the question.
        until = sample.question_time if sample.question_time is not None else float("inf")
        first = None
        for c in merged:
            path = Path(c["path"])
            first = first or str(path)
            if float(c["offset"]) > until or (path.is_file() and path.stat().st_size > 0):
                continue
            if not c.get("drive_file_id"):
                raise FileNotFoundError(f"Clip missing at {path}, and no drive_file_id for it.")
            download_drive_file(path, c["drive_file_id"])
        return first

    video_path = Path(sample.video)
    if video_path.is_file() and video_path.stat().st_size > 0:
        return str(video_path)
    drive_id = meta.get("drive_file_id")
    if not drive_id:
        raise FileNotFoundError(f"Video missing at {video_path}, and no drive_file_id in sample metadata.")
    download_drive_file(video_path, drive_id)
    return str(video_path)
