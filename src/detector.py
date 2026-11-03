"""Face detection with YuNet, plus a multi-scale tiled scan for small faces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
TILE = 640  # YuNet input size used for each tile


@dataclass
class Face:
    """One detected face: box in pixel coordinates (x, y, w, h) and a confidence score."""

    x: float
    y: float
    w: float
    h: float
    score: float

    def as_dict(self) -> dict:
        return {
            "x": round(self.x),
            "y": round(self.y),
            "w": round(self.w),
            "h": round(self.h),
            "score": round(self.score, 3),
        }


def enhance(image: np.ndarray) -> np.ndarray:
    """Brighten dark frames: CLAHE on the lightness channel plus a gamma lift."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    table = (np.linspace(0, 1, 256) ** 0.7 * 255).astype(np.uint8)
    return cv2.LUT(out, table)


class FaceDetector:
    """Wraps OpenCV's YuNet detector and adds tiling, zoom passes and NMS."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
    ) -> None:
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self._net = cv2.FaceDetectorYN.create(
            str(model_path), "", (TILE, TILE), score_threshold, nms_threshold, 5000
        )

    def _detect_tile(self, tile: np.ndarray) -> np.ndarray:
        """Run YuNet on one tile. Returns an (N, 5) array of x, y, w, h, score."""
        h, w = tile.shape[:2]
        self._net.setInputSize((w, h))
        self._net.setScoreThreshold(self.score_threshold)
        _, faces = self._net.detect(tile)
        if faces is None:
            return np.zeros((0, 5), dtype=np.float32)
        return faces[:, [0, 1, 2, 3, 14]].astype(np.float32)

    def _scan(self, image: np.ndarray, scale: float) -> list[np.ndarray]:
        """Scan the image at one scale, in overlapping tiles, and map boxes back."""
        if scale != 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]
        step = int(TILE * 0.75)  # 25% overlap so faces on a tile edge are not cut in half
        ys = list(range(0, max(h - TILE, 0) + 1, step)) or [0]
        xs = list(range(0, max(w - TILE, 0) + 1, step)) or [0]
        if ys[-1] + TILE < h:
            ys.append(h - TILE)
        if xs[-1] + TILE < w:
            xs.append(w - TILE)

        found = []
        for y0 in ys:
            for x0 in xs:
                tile = image[max(y0, 0) : y0 + TILE, max(x0, 0) : x0 + TILE]
                boxes = self._detect_tile(tile)
                if len(boxes):
                    boxes[:, 0] += max(x0, 0)
                    boxes[:, 1] += max(y0, 0)
                    boxes[:, :4] /= scale
                    found.append(boxes)
        return found

    def detect(
        self,
        image: np.ndarray,
        scales: tuple[float, ...] = (1.0, 2.0),
        enhance_low_light: bool = False,
    ) -> list[Face]:
        """Detect faces in a BGR image.

        Each scale above 1 enlarges the image before scanning, which helps find
        faces only a few pixels wide in crowds and CCTV frames. With
        enhance_low_light, contrast is boosted before detection (the redaction
        itself is still applied to the original image).
        """
        if enhance_low_light:
            image = enhance(image)
        found: list[np.ndarray] = []
        for scale in scales:
            found.extend(self._scan(image, scale))
        if not found:
            return []

        boxes = np.concatenate(found)
        keep = cv2.dnn.NMSBoxes(
            boxes[:, :4].tolist(), boxes[:, 4].tolist(), self.score_threshold, self.nms_threshold
        )
        keep = np.array(keep).flatten() if len(keep) else []
        faces = [Face(*boxes[i, :4].tolist(), float(boxes[i, 4])) for i in keep]
        faces.sort(key=lambda f: (f.y, f.x))
        return faces
