"""Per-clip cache of the frozen write-path feature g_t and of the 1-fps frames (docs/plan.md §4, Stage 0).

Layout under <root>, written by main/cli/cache_features.py:
  g/<clip_uid>.npy          [n, D] fp32, g_t = mean over the frame's visual tokens (Phi_v, tower + projector)
  frames/<clip_uid>.bin     the n frames (already resized to frame_max_side) as lossless PNG, concatenated
  frames/<clip_uid>.off.npy [n + 1] int64 byte offsets into the .bin
  meta/<clip_uid>.json      written last; with "error": null and "frame_format": "png" it marks the clip complete

Frame k of a clip is the k-th frame `iter_video_frames` yields, capped at ceil(duration * fps) as in
`iter_stream_frames`, so a stream built from the cache has the same frames, times and cuts as the
online path. The write path reads only g_t (fp16 tower, cosine to the online value >= 0.9999); the frame
shown to the decoder at a query is pixel-identical (PNG). A JPEG q95 copy was tried and rejected: it moved
the prompt hidden state by 6.8%, more than one real frame step (5.4%), main/cli/verify_cache.py.
"""
from __future__ import annotations

import io
import json
import math
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


class LazyFrame:
    """A cached frame decoded only when a query needs it."""

    def __init__(self, cache: "FrameCache", clip_uid: str, k: int):
        self.cache, self.clip_uid, self.k = cache, clip_uid, k

    def get(self) -> Image.Image:
        return self.cache.image(self.clip_uid, self.k)


def as_image(frame) -> Image.Image:
    return frame.get() if isinstance(frame, LazyFrame) else frame


class FrameCache:
    """`video_dir` is the fallback for a cache shipped without frames/ (only g/ + meta/): frame k is then
    decoded from <video_dir>/<clip_uid>.mp4 by the same `iter_video_frames` call cache_features used, so it
    matches the PNG when the OpenCV build matches. Frames are asked for in increasing k within a clip, so one
    sequential decoder per clip is kept open and advanced; an earlier k restarts it."""

    def __init__(self, root: str, video_dir: Optional[str] = None, sample_fps: float = 1.0, max_side: Optional[int] = 448):
        self.root = root
        self.video_dir, self.sample_fps, self.max_side = video_dir, sample_fps, max_side
        self._meta: Dict[str, dict] = {}
        self._g: Dict[str, np.ndarray] = {}
        self._dec: Optional[Tuple[str, Iterator, int, Optional[Image.Image]]] = None  # (clip, iterator, next k, last frame)

    def meta(self, clip_uid: str) -> dict:
        if clip_uid not in self._meta:
            path = os.path.join(self.root, "meta", f"{clip_uid}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"clip {clip_uid} is not in the feature cache {self.root}")
            with open(path) as f:
                m = json.load(f)
            if m.get("error"):
                raise RuntimeError(f"clip {clip_uid} failed in the feature cache: {m['error']}")
            if m.get("frame_format") != "png":
                raise RuntimeError(f"clip {clip_uid} is cached in an old frame format ({m.get('frame_format', 'jpeg')}); re-run cache_features")
            self._meta[clip_uid] = m
        return self._meta[clip_uid]

    def has(self, clip_uid: str) -> bool:
        try:
            self.meta(clip_uid)
            return True
        except (FileNotFoundError, RuntimeError):
            return False

    def g(self, clip_uid: str) -> np.ndarray:
        if clip_uid not in self._g:
            if len(self._g) > 16:  # streams have <= 7 clips; keep memory flat
                self._g.clear()
            self._g[clip_uid] = np.load(os.path.join(self.root, "g", f"{clip_uid}.npy"), mmap_mode="r")
        return self._g[clip_uid]

    def feature_mean(self, clip_uids) -> np.ndarray:
        """Mean g_t over every cached frame of the given clips, fp32 [D] (accumulated in fp64)."""
        total, n = None, 0
        for u in clip_uids:
            g = np.load(os.path.join(self.root, "g", f"{u}.npy"), mmap_mode="r")
            s = np.asarray(g, dtype=np.float64).sum(0)
            total = s if total is None else total + s
            n += len(g)
        if not n:
            raise ValueError("feature_mean: no cached frames")
        return (total / n).astype(np.float32)

    def num_frames(self, clip_uid: str) -> int:
        return int(self.meta(clip_uid)["n_frames"])

    def image(self, clip_uid: str, k: int) -> Image.Image:
        if not os.path.exists(os.path.join(self.root, "frames", f"{clip_uid}.bin")):
            return self._decode(clip_uid, k)
        off = np.load(os.path.join(self.root, "frames", f"{clip_uid}.off.npy"))
        with open(os.path.join(self.root, "frames", f"{clip_uid}.bin"), "rb") as f:
            f.seek(int(off[k]))
            buf = f.read(int(off[k + 1] - off[k]))
        return Image.open(io.BytesIO(buf)).convert("RGB")

    def _decode(self, clip_uid: str, k: int) -> Image.Image:
        if self.video_dir is None:
            raise FileNotFoundError(f"clip {clip_uid}: no cached frames in {self.root}/frames and no video_dir to decode from")
        from main.utils.video import iter_video_frames

        if self._dec is None or self._dec[0] != clip_uid or k < self._dec[2] - 1:
            path = os.path.join(self.video_dir, f"{clip_uid}.mp4")
            self._dec = (clip_uid, iter_video_frames(path, sample_fps=self.sample_fps, max_side=self.max_side), 0, None)
        uid, it, nxt, last = self._dec
        if k == nxt - 1 and last is not None:  # the same frame asked again (several questions at one t_q)
            return last
        while nxt <= k:
            try:
                last, _ = next(it)
            except StopIteration:
                raise IndexError(f"clip {clip_uid}: frame {k} is past the end of {self.video_dir}/{clip_uid}.mp4 ({nxt} frames)")
            nxt += 1
        self._dec = (uid, it, nxt, last)
        return last

    def stream_index(self, item, sample_fps: float) -> List[Tuple[str, int, float]]:
        """(clip_uid, k, video time) for every frame the stream shows; mirrors iter_stream_frames."""
        seq: List[Tuple[str, int, float]] = []
        last = -math.inf
        for p in item.pieces:
            n = min(int(math.ceil(p.duration * sample_fps - 1e-6)), self.num_frames(p.clip_uid))
            for k in range(n):
                t = p.video_start + k / sample_fps
                if t > last + 0.5 / sample_fps:
                    last = t
                    seq.append((p.clip_uid, k, t))
        return seq

    def stream(self, item, sample_fps: float) -> Iterator[Tuple[LazyFrame, torch.Tensor, float, bool, bool]]:
        """(frame, g_t [D] fp32 cpu, t, cut, last), with `cut` exactly as train_ego4d.stream_frames."""
        seq = self.stream_index(item, sample_fps)
        gap = 1.5 / sample_fps
        for i, (uid, k, t) in enumerate(seq):
            last = i == len(seq) - 1
            cut = last or seq[i + 1][2] - t > gap
            yield LazyFrame(self, uid, k), torch.from_numpy(np.array(self.g(uid)[k], dtype=np.float32)), t, cut, last

    def stream_g(self, item, sample_fps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Whole stream at once for offline analysis: g [T, D] fp32, times [T], cut flags [T]."""
        seq = self.stream_index(item, sample_fps)
        g = np.stack([np.asarray(self.g(u)[k], dtype=np.float32) for u, k, _ in seq]) if seq else np.zeros((0, 0), np.float32)
        t = np.array([s[2] for s in seq], dtype=np.float64)
        cut = np.zeros(len(seq), dtype=bool)
        if len(seq):
            cut[:-1] = np.diff(t) > 1.5 / sample_fps
            cut[-1] = True
        return g, t, cut


def stream_inputs(item, sample_fps: float, max_side: int, cache: Optional[FrameCache]):
    """(frame, g or None, t, cut, last) for one stream: from the cache when given, else decoded online."""
    if cache is not None:
        yield from cache.stream(item, sample_fps)
        return
    from main.cli.train_ego4d import stream_frames

    for img, t, cut, last in stream_frames(item, sample_fps, max_side):
        yield img, None, t, cut, last
