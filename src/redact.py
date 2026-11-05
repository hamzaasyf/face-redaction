"""Redaction styles: pixelate, blur or solid block, with optional oval mask."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

STYLES = ("pixelate", "blur", "solid")


def _padded_box(x, y, w, h, padding, img_w, img_h):
    """Grow a box by `padding` (fraction of its size) and clip it to the image."""
    px, py = w * padding, h * padding
    x0 = int(max(0, x - px))
    y0 = int(max(0, y - py))
    x1 = int(min(img_w, x + w + px))
    y1 = int(min(img_h, y + h + py))
    return x0, y0, x1, y1


def _apply_style(region: np.ndarray, style: str, strength: int) -> np.ndarray:
    h, w = region.shape[:2]
    if style == "pixelate":
        # strength = how many blocks across the face; fewer blocks hide more
        blocks = max(2, 30 - strength)
        small = cv2.resize(region, (max(1, w * blocks // max(w, h)), max(1, h * blocks // max(w, h))),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if style == "blur":
        k = max(3, int(min(w, h) * strength / 25) | 1)  # odd kernel size, scales with face size
        return cv2.GaussianBlur(region, (k, k), 0)
    if style == "solid":
        return np.zeros_like(region)
    raise ValueError(f"Unknown style: {style}. Use one of {STYLES}.")


def redact(
    image: np.ndarray,
    boxes: Iterable[tuple[float, float, float, float]],
    style: str = "pixelate",
    strength: int = 18,
    padding: float = 0.25,
    oval: bool = True,
) -> np.ndarray:
    """Return a copy of `image` with every box redacted.

    boxes: (x, y, w, h) tuples in pixels.
    strength: 1 to 28. Higher hides more.
    padding: extra margin around each face, as a fraction of its size.
    oval: follow the face shape instead of a rectangle.
    """
    out = image.copy()
    img_h, img_w = image.shape[:2]
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = _padded_box(x, y, w, h, padding, img_w, img_h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = out[y0:y1, x0:x1]
        hidden = _apply_style(region, style, strength)
        if oval:
            mask = np.zeros(region.shape[:2], dtype=np.uint8)
            cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), ((x1 - x0) // 2, (y1 - y0) // 2),
                        0, 0, 360, 255, -1)
            mask3 = cv2.merge([mask, mask, mask]) > 0
            region[mask3] = hidden[mask3]
        else:
            out[y0:y1, x0:x1] = hidden
    return out


def _corners(img, p0, p1, color, thickness):
    """Draw only the four corners of a box, like a camera focus frame."""
    (x0, y0), (x1, y1) = p0, p1
    L = max(4, int(min(x1 - x0, y1 - y0) * 0.25))
    for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, thickness, cv2.LINE_AA)


def draw_boxes(image: np.ndarray, boxes, labels=None, show_numbers: bool = False,
               color=(100, 220, 80), spared_color=(40, 170, 255)) -> np.ndarray:
    """Draw thin corner-bracket markers around faces.

    Small images are enlarged first so the lines stay thin and crisp on screen.
    labels: optional list of (number, score, spared) tuples, one per box.
    show_numbers: add a "#n score" tag on the larger faces. Spared faces are
    always tagged. Colours are BGR: green for redacted faces, amber for spared.
    """
    up = max(1.0, 1600 / max(image.shape[:2]))
    out = cv2.resize(image, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC) if up > 1 else image.copy()
    boxes = [(x * up, y * up, w * up, h * up) for x, y, w, h in boxes]
    scale = max(out.shape[:2]) / 900
    thickness = max(1, round(1.1 * scale))
    font_scale = max(0.35, 0.42 * scale)
    for i, (x, y, w, h) in enumerate(boxes):
        num, score, spared = labels[i] if labels is not None else (i + 1, None, False)
        c = spared_color if spared else color
        p0, p1 = (int(x), int(y)), (int(x + w), int(y + h))
        _corners(out, p0, p1, c, thickness)
        if not spared and (not show_numbers or w < 45 * scale):
            continue  # corners only, unless numbers are switched on
        text = f"#{num}" + (f" {score:.2f}" if score is not None else "") + (" spared" if spared else "")
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = p0[1] - 4 if p0[1] - th - base - 6 > 0 else p1[1] + th + base + 4
        tx = min(p0[0], out.shape[1] - tw - 7)  # keep the tag inside the image
        cv2.rectangle(out, (tx, ty - th - base - 2), (tx + tw + 6, ty + 2), c, -1)
        cv2.putText(out, text, (tx + 3, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return out
