"""Command-line use: redact a photo or a video without the web interface.

    python cli.py photo.jpg -o redacted.png
    python cli.py clip.mp4 -o redacted.mp4 --style blur
"""

import argparse
import json
from pathlib import Path

import cv2

from src.detector import FaceDetector
from src.redact import STYLES, redact
from src.video import redact_video, save_report

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Redact every face in a photo or video.")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--style", choices=STYLES, default="pixelate")
    ap.add_argument("--strength", type=int, default=18)
    ap.add_argument("--padding", type=float, default=0.25)
    ap.add_argument("--rect", action="store_true", help="Rectangular mask instead of oval")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--scales", default="1,2", help="Zoom passes, e.g. 1 or 1,2 or 1,2,3")
    ap.add_argument("--low-light", action="store_true", help="Boost dark frames before detection")
    args = ap.parse_args()

    scales = tuple(float(s) for s in args.scales.split(","))
    detector = FaceDetector(score_threshold=args.threshold)

    if Path(args.input).suffix.lower() in VIDEO_EXT:
        report = redact_video(args.input, args.output, detector, args.style, args.strength,
                              args.padding, not args.rect, scales)
        save_report(report, Path(args.output).with_suffix(".json"))
        print(f"{report['frames']} frames, {report['faces_tracked']} faces tracked -> {args.output}")
    else:
        image = cv2.imread(args.input)
        if image is None:
            raise SystemExit(f"Could not read {args.input}")
        faces = detector.detect(image, scales=scales, enhance_low_light=args.low_light)
        out = redact(image, [(f.x, f.y, f.w, f.h) for f in faces], args.style, args.strength,
                     args.padding, not args.rect)
        cv2.imwrite(args.output, out)
        print(json.dumps({"faces_found": len(faces), "output": args.output}))


if __name__ == "__main__":
    main()
