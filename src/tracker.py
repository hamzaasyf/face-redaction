"""A small IoU tracker that keeps faces covered when the detector misses a frame.

Matching follows the ByteTrack idea in two stages: confident detections are
matched to tracks first, then low-confidence detections are used to rescue
tracks that would otherwise be lost. Each track keeps a constant-velocity
estimate, so for a few frames after its last detection its predicted box is
still redacted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4), boxes as x, y, w, h."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax0, ay0 = a[:, 0:1], a[:, 1:2]
    ax1, ay1 = ax0 + a[:, 2:3], ay0 + a[:, 3:4]
    bx0, by0 = b[:, 0], b[:, 1]
    bx1, by1 = bx0 + b[:, 2], by0 + b[:, 3]
    iw = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
    ih = np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-6)


def _greedy_match(scores: np.ndarray, threshold: float):
    """Match rows to columns greedily by highest score above threshold."""
    matches, used_r, used_c = [], set(), set()
    if scores.size == 0:
        return matches
    order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
    for r, c in order:
        if scores[r, c] < threshold:
            break
        if r in used_r or c in used_c:
            continue
        matches.append((int(r), int(c)))
        used_r.add(r)
        used_c.add(c)
    return matches


@dataclass
class Track:
    track_id: int
    box: np.ndarray  # x, y, w, h
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    missed: int = 0
    hits: int = 1
    frames_detected: int = 1
    frames_predicted: int = 0

    def predict(self) -> np.ndarray:
        return self.box + self.velocity

    def update(self, box: np.ndarray) -> None:
        self.velocity = 0.6 * self.velocity + 0.4 * (box - self.box)
        self.box = box
        self.missed = 0
        self.hits += 1
        self.frames_detected += 1


class FaceTracker:
    """Tracks faces across frames and returns the boxes to redact for each frame."""

    def __init__(self, high_score: float = 0.6, iou_threshold: float = 0.3, max_missed: int = 8):
        self.high_score = high_score
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed  # frames to keep covering a face after the detector loses it
        self.tracks: list[Track] = []
        self._next_id = 1
        self.finished: list[Track] = []

    def update(self, detections: np.ndarray) -> list[tuple[int, np.ndarray, bool]]:
        """detections: (N, 5) array of x, y, w, h, score.

        Returns a list of (track_id, box, predicted) for this frame. `predicted`
        is True when the box comes from the tracker rather than a detection.
        """
        high = detections[detections[:, 4] >= self.high_score] if len(detections) else detections
        low = detections[detections[:, 4] < self.high_score] if len(detections) else detections

        predictions = np.array([t.predict() for t in self.tracks]) if self.tracks else np.zeros((0, 4))
        unmatched_tracks = set(range(len(self.tracks)))

        # Stage 1: confident detections
        unmatched_high = set(range(len(high)))
        for ti, di in _greedy_match(iou(predictions, high[:, :4]), self.iou_threshold):
            self.tracks[ti].update(high[di, :4])
            unmatched_tracks.discard(ti)
            unmatched_high.discard(di)

        # Stage 2: low-confidence detections rescue remaining tracks
        remaining = sorted(unmatched_tracks)
        if remaining and len(low):
            scores = iou(predictions[remaining], low[:, :4])
            for ri, di in _greedy_match(scores, self.iou_threshold):
                ti = remaining[ri]
                self.tracks[ti].update(low[di, :4])
                unmatched_tracks.discard(ti)

        # Tracks with no match this frame coast on their predicted position
        for ti in unmatched_tracks:
            t = self.tracks[ti]
            t.box = t.predict()
            t.missed += 1
            t.frames_predicted += 1

        # New tracks for confident detections that matched nothing
        for di in unmatched_high:
            self.tracks.append(Track(self._next_id, high[di, :4].copy()))
            self._next_id += 1

        # Drop tracks lost for too long
        alive = []
        for t in self.tracks:
            (alive if t.missed <= self.max_missed else self.finished).append(t)
        self.tracks = alive

        return [(t.track_id, t.box.copy(), t.missed > 0) for t in self.tracks]

    def summary(self) -> list[dict]:
        """Per-track counts for the JSON report."""
        everything = sorted(self.finished + self.tracks, key=lambda t: t.track_id)
        return [
            {
                "track_id": t.track_id,
                "frames_detected": t.frames_detected,
                "frames_covered_by_tracker": t.frames_predicted,
            }
            for t in everything
        ]
