"""Ego4D NLQ / QaEgo4D stream-level dataset over `dataset/clean` (format: dataset/README.md).

One item = one source video replayed as a stream: all its annotated clips of the split, in video
order, at their video times (gaps between clips are skipped, overlapping frames are shown once).
Each query carries an ask time t_q, a frame time at or after its evidence end, so the stream is
played once and every query is read mid-stream at its own t_q (docs/pilot.md §10).
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence
from torch.utils.data import Dataset
SOURCES = ("merged", "nlq", "qaego4d")
QUERY_TIME_MODES = ("uniform", "final")
@dataclass(frozen=True)
class Query:
    query_id: str
    source: str
    clip_uid: str
    query: str
    answer: Optional[str]
    window_start: float  # stream (= video) seconds
    window_end: float
    template: Optional[str]
    slot_x: Optional[str]
    nlq_query_id: Optional[str]
    ask_time: float = -1.0
    @property
    def key(self) -> str:
        # Linked NLQ/QaEgo4D pairs share this key, so both get the same ask time.
        return self.nlq_query_id or self.query_id
    @property
    def is_linked(self) -> bool:
        return self.source == "qaego4d" and self.nlq_query_id is not None
@dataclass(frozen=True)
class ClipPiece:
    clip_uid: str
    path: str
    video_start: float  # clip offset in the video (s)
    duration: float
@dataclass(frozen=True)
class StreamItem:
    stream_id: str  # = video_uid
    split: str
    participant_id: Optional[int]
    pieces: List[ClipPiece]  # in video order
    frame_times: List[float]  # nominal frame times (video s) the stream shows, overlap removed
    queries: List[Query]
    @property
    def clip_uids(self) -> List[str]:
        return [p.clip_uid for p in self.pieces]
    @property
    def num_frames(self) -> int:
        return len(self.frame_times)
def stream_frame_times(pieces: Sequence[ClipPiece], sample_fps: float) -> List[float]:
    """Frame k of a clip is at video time start + k / fps; a frame is dropped when an earlier clip
    already showed that moment (overlapping clips). Mirrors streamlite.utils.video.iter_stream_frames."""
    out: List[float] = []
    last = -math.inf
    for p in pieces:
        n = int(math.ceil(p.duration * sample_fps - 1e-6))
        for k in range(n):
            t = p.video_start + k / sample_fps
            if t > last + 0.5 / sample_fps:
                out.append(t)
                last = t
    return out
def _stable_unit(*parts) -> float:
    # Python's hash() is salted per process; ask times must be identical across runs and arms.
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64
def sample_ask_time(window_end: float, frame_times: Sequence[float], mode: str, u: float) -> float:
    """t_q drawn uniformly over the stream frames shown at or after the evidence end (u ~ U(0,1)).

    No minimum delay: t_q may fall while the evidence is still in the open segment or long after it
    was committed. If the window ends after the last frame, t_q is the last frame.
    """
    if mode == "final":
        return frame_times[-1]
    # first frame index with t >= window_end (frame_times is sorted)
    lo, hi = 0, len(frame_times)
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_times[mid] < window_end - 1e-6:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(frame_times):
        return frame_times[-1]
    return frame_times[lo + min(int(u * (len(frame_times) - lo)), len(frame_times) - lo - 1)]
def _row_to_query(r: Dict, offset: float) -> Query:
    return Query(
        query_id=r["query_id"],
        source=r["source"],
        clip_uid=r["clip_uid"],
        query=r["query"],
        answer=r["answer"],
        window_start=offset + float(r["clip_start_sec"]),
        window_end=offset + float(r["clip_end_sec"]),
        template=r["template"],
        slot_x=r["slot_x"],
        nlq_query_id=r["nlq_query_id"],
    )
def load_queries(clean_dir: str, split: str, source: str, offsets: Dict[str, float]) -> List[Query]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source == "merged" and split != "train":
        raise ValueError("Merged NLQ+QaEgo4D queries are for training only; evaluate one track at a time.")
    def read(name: str) -> List[Query]:
        with open(os.path.join(clean_dir, f"{name}_{split}.json")) as f:
            return [_row_to_query(r, offsets[r["clip_uid"]]) for r in json.load(f)]
    if source != "merged":
        return read(source)
    # A linked pair is one question: keep the QaEgo4D row (it has the answer), drop its NLQ twin.
    qa = read("qaego4d")
    linked = {q.nlq_query_id for q in qa if q.nlq_query_id is not None}
    nlq = [q for q in read("nlq") if q.query_id not in linked]
    return qa + nlq
def dev_participants(clips: Sequence[Dict], frac: float, seed: int) -> set:
    # Unknown participants are never put in dev: their disjointness from train can't be shown.
    pids = sorted({c["participant_id"] for c in clips if c["split"] == "train" and c["participant_id"] is not None})
    random.Random(seed).shuffle(pids)
    return set(pids[: int(round(frac * len(pids)))])
class Ego4DStreamDataset(Dataset):
    def __init__(
        self,
        clean_dir: str,
        video_dir: str,
        split: str,
        source: str,
        part: str = "all",
        dev_frac: float = 0.1,
        query_time_mode: str = "uniform",
        sample_fps: float = 1.0,
        resample_ask_times: bool = False,
        seed: int = 42,
        require_video: bool = True,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"split must be train or val (test has no annotations), got {split!r}")
        if part not in ("all", "train", "dev"):
            raise ValueError(f"part must be all|train|dev, got {part!r}")
        if part != "all" and split != "train":
            raise ValueError("`part` carves dev out of the train split only.")
        if query_time_mode not in QUERY_TIME_MODES:
            raise ValueError(f"query_time_mode must be one of {QUERY_TIME_MODES}, got {query_time_mode!r}")
        self.query_time_mode = query_time_mode
        self.sample_fps = float(sample_fps)
        self.resample_ask_times = bool(resample_ask_times)
        self.seed = int(seed)
        self.epoch = 0
        with open(os.path.join(clean_dir, "clips.json")) as f:
            all_clips = json.load(f)
        dev = dev_participants(all_clips, dev_frac, seed) if part != "all" else set()
        offsets = {c["clip_uid"]: float(c["video_start_sec"]) for c in all_clips}
        by_video: Dict[str, List[Query]] = {}
        clip_video = {c["clip_uid"]: c["video_uid"] for c in all_clips}
        for q in load_queries(clean_dir, split, source, offsets):
            by_video.setdefault(clip_video[q.clip_uid], []).append(q)
        clips_by_video: Dict[str, List[Dict]] = {}
        for c in all_clips:
            if c["split"] == split:
                clips_by_video.setdefault(c["video_uid"], []).append(c)
        self.items: List[StreamItem] = []
        self.missing_video: List[str] = []
        for vid in sorted(clips_by_video):
            qs = by_video.get(vid, [])
            if not qs:
                continue
            cs = sorted(clips_by_video[vid], key=lambda c: c["video_start_sec"])
            pid = cs[0]["participant_id"]
            in_dev = pid in dev
            if (part == "train" and in_dev) or (part == "dev" and not in_dev):
                continue
            pieces = [ClipPiece(c["clip_uid"], os.path.join(video_dir, f"{c['clip_uid']}.mp4"), float(c["video_start_sec"]), float(c["duration_sec"])) for c in cs]
            if require_video and not all(os.path.exists(p.path) for p in pieces):
                self.missing_video.append(vid)  # a stream with a hole would be a different stream
                continue
            self.items.append(StreamItem(vid, split, pid, pieces, stream_frame_times(pieces, self.sample_fps), qs))
        if self.missing_video:
            print(f"[Ego4DStreamDataset] {split}/{source}/{part}: dropped {len(self.missing_video)} streams with a missing clip mp4 in {video_dir}")
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
    def _with_ask_times(self, item: StreamItem) -> StreamItem:
        # Evaluation: ask time depends only on (seed, query key). Training with resampling: also on epoch.
        epoch = self.epoch if self.resample_ask_times else 0
        qs = [replace(q, ask_time=sample_ask_time(q.window_end, item.frame_times, self.query_time_mode, _stable_unit(self.seed, epoch, q.key)))
              for q in item.queries]
        qs.sort(key=lambda q: (q.ask_time, q.query_id))
        return replace(item, queries=qs)
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self, idx: int) -> StreamItem:
        return self._with_ask_times(self.items[idx])
def collate_stream(batch: List[StreamItem]) -> StreamItem:
    if len(batch) != 1:
        raise ValueError(f"One stream per step (the stream is sequential), got batch of {len(batch)}.")
    return batch[0]
