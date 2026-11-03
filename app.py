"""Gradio demo: redact every face in a photo or video."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr

from src.detector import FaceDetector
from src.redact import draw_boxes, redact
from src.video import redact_video, save_report
from ui import CSS, HEAD, THEME, header_html, stats_html, video_header_html

SCAN_MODES = {"1×": (1.0,), "1× + 2×": (1.0, 2.0), "+3× CCTV": (1.0, 2.0, 3.0)}
STYLE_NAMES = {"Pixelate": "pixelate", "Blur": "blur", "Black bar": "solid"}
# Free CPU Spaces are slow, so videos are trimmed unless this is raised
MAX_VIDEO_SECONDS = float(os.environ.get("MAX_VIDEO_SECONDS", "20"))
SAMPLES = Path(__file__).parent / "samples"


def _parse_ids(text: str) -> set[int]:
    return {int(p) for p in (text or "").replace(";", ",").split(",") if p.strip().isdigit()}


def process_photo(image, style, strength, padding, oval, scan, threshold, low_light, spare, show_boxes):
    if image is None:
        raise gr.Error("Add a photo first.")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detector = FaceDetector(score_threshold=threshold)

    started = time.time()
    faces = detector.detect(bgr, scales=SCAN_MODES[scan], enhance_low_light=low_light)
    ms = round((time.time() - started) * 1000)

    spared = {i for i in _parse_ids(spare) if 1 <= i <= len(faces)}
    boxes = [(f.x, f.y, f.w, f.h) for f in faces]
    to_hide = [b for i, b in enumerate(boxes, start=1) if i not in spared]
    clean = redact(bgr, to_hide, STYLE_NAMES[style], int(strength), padding / 100, oval)

    shown = clean
    if show_boxes:
        labels = [(i, f.score, i in spared) for i, f in enumerate(faces, start=1)]
        shown = draw_boxes(clean, boxes, labels)

    out_path = Path(tempfile.mkdtemp()) / "redacted.png"
    cv2.imwrite(str(out_path), clean)

    report = {
        "faces_found": len(faces),
        "faces_redacted": len(to_hide),
        "faces_spared": sorted(spared),
        "detection_ms": ms,
        "scan_scales": list(SCAN_MODES[scan]),
        "faces": [{"id": i, **f.as_dict()} for i, f in enumerate(faces, start=1)],
    }
    h, w = bgr.shape[:2]
    return (
        cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
        stats_html(len(faces), len(to_hide), len(spared), f"{ms} ms"),
        str(out_path),
        report,
        header_html(f"{w}×{h}", stamped=True),
    )


def process_video(video, style, strength, padding, oval, scan, threshold, keep_frames,
                  progress=gr.Progress()):
    if not video:
        raise gr.Error("Add a video first.")
    detector = FaceDetector(score_threshold=threshold)
    out_dir = Path(tempfile.mkdtemp())
    out_path = out_dir / "redacted.mp4"
    report = redact_video(
        video, out_path, detector,
        style=STYLE_NAMES[style], strength=int(strength), padding=padding / 100, oval=oval,
        scales=SCAN_MODES[scan], max_missed=int(keep_frames),
        max_seconds=MAX_VIDEO_SECONDS, progress=lambda p: progress(p, desc="Redacting frames"),
    )
    report_path = out_dir / "report.json"
    save_report(report, report_path)
    return (
        str(out_path),
        video_header_html(report["frames"], report["faces_tracked"]),
        stats_html(report["faces_tracked"], report["boxes_from_detector"], report["boxes_from_tracker"],
                   f"{report['processing_seconds']} s",
                   labels=("faces tracked", "boxes detected", "held by tracker", "processing")),
        str(report_path),
    )


def redaction_controls():
    style = gr.Radio(list(STYLE_NAMES), value="Pixelate", label="Method", elem_classes="seg")
    strength = gr.Slider(1, 28, value=18, step=1, label="Strength")
    padding = gr.Slider(0, 60, value=25, step=5, label="Margin around face (%)")
    oval = gr.Checkbox(value=True, label="Oval mask, follows the face shape")
    return style, strength, padding, oval


with gr.Blocks(title="Face Redaction") as demo:
    header = gr.HTML(header_html())

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("Photo"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes="paper"):
                    photo_in = gr.Image(type="numpy", label="Photo", height=220, elem_classes="drop")
                    p_style, p_strength, p_padding, p_oval = redaction_controls()
                    p_scan = gr.Radio(list(SCAN_MODES), value="1× + 2×", label="Scan depth", elem_classes="seg")
                    with gr.Accordion("More settings", open=False, elem_classes="more"):
                        p_threshold = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Confidence threshold")
                        p_low_light = gr.Checkbox(value=False, label="Low-light boost for dark CCTV frames")
                        p_boxes = gr.Checkbox(value=True, label="Show detection boxes on the result")
                    p_spare = gr.Textbox(label="Exempt faces", placeholder="Face numbers to leave visible, e.g. 2, 5")
                    photo_btn = gr.Button("Redact faces", variant="primary", elem_classes="go")
                    gr.HTML('<p class="note">Photos are processed for this request only and are not stored.</p>')
                with gr.Column(scale=8, elem_classes="exhibit"):
                    photo_out = gr.Image(label="Fig. A — after", interactive=False, height=520,
                                         elem_classes="figure", buttons=["fullscreen"])
                    photo_stats = gr.HTML(stats_html())
                    photo_file = gr.File(label="Download clean PNG (no boxes)", elem_classes="paperfile")
                    with gr.Accordion("Report (JSON)", open=False, elem_classes="more dark"):
                        photo_json = gr.JSON(show_label=False)

            examples = sorted(str(p) for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if examples:
                gr.Examples(examples=examples, inputs=photo_in, label="Sample exhibits")

            photo_btn.click(
                process_photo,
                [photo_in, p_style, p_strength, p_padding, p_oval, p_scan, p_threshold, p_low_light, p_spare, p_boxes],
                [photo_out, photo_stats, photo_file, photo_json, header],
            )

        with gr.Tab("Video"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes="paper"):
                    video_in = gr.Video(label="Video", height=220, elem_classes="drop")
                    v_style, v_strength, v_padding, v_oval = redaction_controls()
                    v_scan = gr.Radio(list(SCAN_MODES), value="1× + 2×", label="Scan depth", elem_classes="seg")
                    with gr.Accordion("More settings", open=False, elem_classes="more"):
                        v_threshold = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="Confidence threshold")
                        v_keep = gr.Slider(0, 30, value=8, step=1, label="Keep covering a lost face for (frames)")
                    video_btn = gr.Button("Redact video", variant="primary", elem_classes="go")
                    gr.HTML(f'<p class="note">This demo processes the first {MAX_VIDEO_SECONDS:g} seconds of each video.</p>')
                with gr.Column(scale=8, elem_classes="exhibit"):
                    cam = gr.HTML(video_header_html())
                    video_out = gr.Video(label="Footage — after", interactive=False, height=480, elem_classes="monitor")
                    video_stats = gr.HTML(stats_html(labels=("faces tracked", "boxes detected", "held by tracker", "processing")))
                    video_report = gr.File(label="Tracking report (JSON)", elem_classes="paperfile")

            video_btn.click(
                process_video,
                [video_in, v_style, v_strength, v_padding, v_oval, v_scan, v_threshold, v_keep],
                [video_out, cam, video_stats, video_report],
            )

    gr.HTML(
        '<footer class="foot">Built by <a href="https://www.linkedin.com/in/hamzaasyf/">Muhammad Hamza Asif</a>'
        ' · <a href="https://github.com/hamzaasyf/face-redaction">Source on GitHub</a>'
        ' · YuNet face detector (OpenCV Zoo)</footer>'
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, head=HEAD)
