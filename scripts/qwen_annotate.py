#!/usr/bin/env python3
"""
Fills the annotation gaps the dataset visualiser flags (num_people,
semantic_location, country-relevant cues) by sampling frames from each
staged clip and asking a Qwen-VL model, served locally by Ollama, for a
structured read of each frame.

Run by scripts/ada_qwen_annotate.sbatch on Ada, after Ollama is serving
on localhost:11434 and clips are staged in --clips-dir. Not meant to run
on this laptop (no video decode libs installed locally, and all GPU/ML
work for this project runs on Ada per project convention).
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib import request

PROMPT = """You are labelling a single frame from a first-person (egocentric) video for a
research dataset. Reply with ONLY a JSON object, no prose, matching this schema:
{"num_people_visible": <int, excluding the camera wearer>,
 "location_guess": "<short phrase, e.g. 'kitchen', 'workshop', 'street'>",
 "indoor_outdoor": "indoor" | "outdoor" | "unclear",
 "confidence": "low" | "medium" | "high"}"""


def extract_frames(clip_path: Path, n: int, out_dir: Path) -> list[Path]:
    """Evenly spaced frames via ffmpeg (assumed present on Ada compute nodes)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 0.0
    frames = []
    for i in range(n):
        t = (duration * (i + 0.5) / n) if duration > 0 else i
        frame_path = out_dir / f"{clip_path.stem}_{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(t), "-i", str(clip_path),
             "-frames:v", "1", "-q:v", "3", str(frame_path)],
            capture_output=True,
        )
        if frame_path.exists():
            frames.append(frame_path)
    return frames


def ask_ollama(model: str, image_path: Path, base_url: str) -> dict:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    payload = json.dumps({
        "model": model, "prompt": PROMPT, "images": [b64],
        "stream": False, "format": "json",
    }).encode()
    req = request.Request(f"{base_url}/api/generate", data=payload,
                           headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    try:
        return json.loads(body["response"])
    except (KeyError, json.JSONDecodeError):
        return {"error": "unparseable_response", "raw": body.get("response")}


def aggregate(frame_results: list[dict]) -> dict:
    valid = [r for r in frame_results if "error" not in r]
    if not valid:
        return {"num_people_visible": None, "location_guess": None,
                "indoor_outdoor": None, "confidence": "low", "n_frames_ok": 0}
    nums = [r.get("num_people_visible") for r in valid if isinstance(r.get("num_people_visible"), int)]
    locs = [r.get("location_guess") for r in valid if r.get("location_guess")]
    ios = [r.get("indoor_outdoor") for r in valid if r.get("indoor_outdoor")]
    return {
        "num_people_visible": max(nums) if nums else None,  # max across frames: a person leaving-frame shouldn't undercount
        "location_guess": max(set(locs), key=locs.count) if locs else None,
        "indoor_outdoor": max(set(ios), key=ios.count) if ios else None,
        "n_frames_ok": len(valid),
        "n_frames_total": len(frame_results),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames-per-clip", type=int, default=6)
    ap.add_argument("--base-url", default="http://localhost:11434")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    clip_uids = sorted({q["clip_uid"] for v in manifest["videos"] for q in v["nlq_queries"]})
    clips_dir = Path(args.clips_dir)

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, clip_uid in enumerate(clip_uids, 1):
            clip_path = clips_dir / f"{clip_uid}.mp4"
            if not clip_path.exists():
                results[clip_uid] = {"error": "clip_not_staged"}
                continue
            print(f"[{i}/{len(clip_uids)}] {clip_uid}", file=sys.stderr)
            frames = extract_frames(clip_path, args.frames_per_clip, tmp_dir)
            if not frames:
                results[clip_uid] = {"error": "frame_extraction_failed"}
                continue
            frame_results = [ask_ollama(args.model, fp, args.base_url) for fp in frames]
            results[clip_uid] = aggregate(frame_results)
            for fp in frames:
                fp.unlink(missing_ok=True)
            # flush after every clip -- a killed/pre-empted job keeps partial progress
            Path(args.out).write_text(json.dumps(
                {"model": args.model, "frames_per_clip": args.frames_per_clip, "annotations": results},
                indent=2))

    print(f"wrote {args.out}: {len(results)} clips", file=sys.stderr)


if __name__ == "__main__":
    main()
