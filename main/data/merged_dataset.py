from __future__ import annotations
import json
import os
from typing import List

from torch.utils.data import Dataset

from main.data.jsonl_dataset import JsonlVLDataset, Sample


class MergedVideoDataset(Dataset):
    """Reads merged_{train,val}.json (one entry per video: its clips in order plus every query).

    Each query becomes one streaming Sample. The stream is the video's clips played back to back; a clip's
    offset is the summed duration of the clips before it, so `combined_*_sec` is the stream clock. The
    question is asked `ask_delay` seconds after its evidence window ends, but never later than the end of
    the clip the question belongs to, so the evidence is already in the observed history. The clip files
    are read from `<clip_dir>/<clip_uid>.mp4`.
    """

    def __init__(self, path: str, clip_dir: str, sample_fps: float = 1.0, ask_delay: float = 1.0,
                 only_answered: bool = True):
        self.items: List[Sample] = []
        with open(path, "r", encoding="utf-8") as f:
            videos = json.load(f)
        for v in videos:
            clips = [
                {
                    "clip_uid": c["clip_uid"],
                    "path": os.path.join(clip_dir, f"{c['clip_uid']}.mp4"),
                    "offset": float(c["clip_offset_sec"]),
                    "duration": float(c["duration_sec"]),
                }
                for c in v["clips"]
            ]
            by_uid = {c["clip_uid"]: c for c in clips}
            for q in v["queries"]:
                if only_answered and not q.get("answer"):
                    continue
                clip = by_uid[q["clip_uid"]]
                start, end = float(q["combined_start_sec"]), float(q["combined_end_sec"])
                question_time = min(end + ask_delay, clip["offset"] + clip["duration"], float(v["combined_duration_sec"]))
                question_time = round(max(question_time, end), 3)
                self.items.append(
                    Sample(
                        id=q["query_id"],
                        image=None,
                        prompt=q["query"],
                        video=f"merged:{v['video_uid']}",
                        answer=q.get("answer"),
                        question_time=question_time,
                        start_time=0.0,
                        end_time=None,
                        sample_fps=sample_fps,
                        max_frames=None,
                        meta={
                            "stream_update_unit": "frame",
                            "evidence_timestamps": [[start, end]],
                            "merged_clips": clips,
                            "combined_duration_sec": float(v["combined_duration_sec"]),
                            "video_uid": v["video_uid"],
                            "clip_uid": q["clip_uid"],
                            "template": q.get("template"),
                        },
                    )
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]


def load_dataset(path: str, clip_dir: str | None = None):
    """`*.jsonl` -> the original per-row dataset; `merged_*.json` -> MergedVideoDataset (needs `clip_dir`)."""
    if path.endswith(".jsonl"):
        return JsonlVLDataset(path)
    if clip_dir is None:
        raise ValueError("Merged datasets need --clip_dir, the folder holding <clip_uid>.mp4 files.")
    return MergedVideoDataset(path, clip_dir)
