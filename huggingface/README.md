---
title: Face Redaction
emoji: 🕶️
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: mit
short_description: Find and hide every face in photos and videos
---

# Face Redaction

Find every face in a photo or video and hide it, while keeping the rest of the image intact. Works on group photos, dense crowds and low-light CCTV frames. In videos, a tracker keeps covering a face for a few frames when the detector misses it, so faces don't flash into view on blurry or turned-away frames.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/hamzaasyf/face-redaction)

![Face Redaction](assets/cover.png)

## Features

- **Photos and videos** in one app, plus a command-line tool
- **Finds small faces:** each frame is scanned at full size, then again on 2x (and 3x) zoomed tiles, so faces a few pixels wide in the back of a crowd are caught
- **Tracking for video:** confident detections start tracks, weaker detections keep existing tracks alive, and each track's position is predicted for frames where the face is missed
- **Three styles:** pixelate, blur or solid block, with adjustable strength, padding and an oval mask that follows the face shape
- **Spare faces:** leave chosen faces visible by number
- **Low-light boost** for dark CCTV frames (applied to detection only, not to the output)
- **Evidence markers:** numbered boxes with confidence scores on the result, plus a clean download without boxes
- **Reports:** JSON with every face box, or per-track counts for videos
- **Audio kept:** redacted videos keep the original soundtrack

![Three redaction styles](assets/styles.png)

## The app

![Face Redaction app](assets/app.png)

## How it works

```
frame ─▶ tiles at 1x / 2x / 3x ─▶ YuNet ─▶ merge (NMS) ─▶ tracker ─▶ redact ─▶ output
                                                            │
                                     predicted boxes for missed frames
```

1. **Detect:** [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) runs on overlapping 640 x 640 tiles at each zoom level.
2. **Merge:** duplicate boxes from overlapping tiles and zoom levels are removed with non-maximum suppression.
3. **Track (video):** two-stage IoU matching in the style of ByteTrack, with a constant-velocity estimate per track.
4. **Redact:** each face is pixelated, blurred or blocked out, and the video is re-encoded to H.264 with the original audio.

## Run locally

```bash
git clone https://github.com/hamzaasyf/face-redaction.git
cd face-redaction
pip install -r requirements.txt
python app.py            # opens the web interface at http://127.0.0.1:7860
```

Command line:

```bash
python cli.py photo.jpg -o redacted.png --style blur
python cli.py clip.mp4 -o redacted.mp4 --scales 1,2
python cli.py cctv.jpg -o redacted.png --scales 1,2,3 --low-light --threshold 0.5
```

## Project structure

```
face-redaction/
├── app.py              Gradio interface (photo and video tabs)
├── cli.py              Command-line tool
├── ui.py               Case-file theme, CSS and header components
├── src/
│   ├── detector.py     YuNet wrapper, tiling, zoom passes, NMS, low-light boost
│   ├── redact.py       Pixelate / blur / solid, oval mask, numbered preview
│   ├── tracker.py      Two-stage IoU tracker with motion prediction
│   └── video.py        Frame loop, tracking, H.264 re-encode with audio
├── models/
│   └── face_detection_yunet_2023mar.onnx
├── samples/            Example photos for the demo
└── requirements.txt
```

## Limitations

- Faces seen from behind, heavily covered, or only a few pixels wide can still be missed. For anything published, review the output, and use the Crowd / CCTV scan and a lower threshold for difficult footage.
- On free CPU hardware, videos are processed at a few frames per second, so the online demo trims videos to the first 20 seconds. Set the `MAX_VIDEO_SECONDS` environment variable to change this.

## Use cases

Privacy compliance for CCTV and body-camera footage, responding to data requests, anonymising training datasets, and publishing event or street footage.

## About

Built by [Muhammad Hamza Asif](https://www.linkedin.com/in/hamzaasyf/), AI engineer working on computer vision, speech AI and LLM applications. Need something like this for your product or workflow? [Get in touch on LinkedIn](https://www.linkedin.com/in/hamzaasyf/).

## Licence

Code: MIT. YuNet model: MIT ([OpenCV Zoo](https://github.com/opencv/opencv_zoo)).
