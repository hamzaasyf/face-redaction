"""Redact every face in a video, with tracking to cover frames the detector misses."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .detector import FaceDetector
from .redact import redact
from .tracker import FaceTracker


def _ffmpeg() -> str | None:
    """Find an ffmpeg binary: system one first, then the one bundled with imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def redact_video(
    src: str | Path,
    dst: str | Path,
    detector: FaceDetector,
    style: str = "blur",
    strength: int = 18,
    padding: float = 0.25,
    oval: bool = True,
    scales: tuple[float, ...] = (1.0,),
    max_missed: int = 8,
    max_seconds: float | None = None,
    progress: Callable[[float], None] | None = None,
) -> dict:
    """Write a redacted H.264 copy of `src` to `dst` (audio kept) and return a report."""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    limit = int(max_seconds * fps) if max_seconds else None
    if limit and total:
        total = min(total, limit)

    tracker = FaceTracker(high_score=detector.score_threshold, max_missed=max_missed)
    # Let the detector return weaker faces too; the tracker uses them to rescue lost tracks
    low_detector_threshold = max(0.3, detector.score_threshold - 0.25)

    tmp_dir = Path(tempfile.mkdtemp())
    silent = tmp_dir / "silent.mp4"
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames, detected_boxes, predicted_boxes = 0, 0, 0
    started = time.time()
    original_threshold = detector.score_threshold
    detector.score_threshold = low_detector_threshold
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (limit and frames >= limit):
                break
            faces = detector.detect(frame, scales=scales)
            dets = np.array([[f.x, f.y, f.w, f.h, f.score] for f in faces], dtype=np.float32).reshape(-1, 5)
            tracked = tracker.update(dets)
            boxes = [tuple(b) for _, b, _ in tracked]
            detected_boxes += sum(1 for *_, p in tracked if not p)
            predicted_boxes += sum(1 for *_, p in tracked if p)
            writer.write(redact(frame, boxes, style, strength, padding, oval))
            frames += 1
            if progress and total:
                progress(min(frames / total, 1.0))
    finally:
        detector.score_threshold = original_threshold
        cap.release()
        writer.release()

    _finalise(src, silent, dst, max_seconds)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "frames": frames,
        "fps": round(fps, 2),
        "resolution": [w, h],
        "processing_seconds": round(time.time() - started, 1),
        "faces_tracked": sum(1 for t in tracker.summary() if t["frames_detected"] >= 3),
        "boxes_from_detector": detected_boxes,
        "boxes_from_tracker": predicted_boxes,
        "tracks": tracker.summary(),
    }


def _finalise(src, silent: Path, dst, max_seconds) -> None:
    """Re-encode to browser-friendly H.264 and copy the original audio, if ffmpeg is available."""
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        shutil.move(str(silent), str(dst))
        return
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(src)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-c:a", "aac", "-shortest", str(dst)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.move(str(silent), str(dst))


def save_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
