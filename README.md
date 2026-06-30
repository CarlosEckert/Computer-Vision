# object-detection-core

Core setup for computer vision projects using YOLO. Works with images, clipboard, screen, webcam, and video.

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) with the `yolov8m-oiv7` model (600 classes from Open Images v7) as the default. 
Intended as a reusable foundation.  
Build on windows and not tested for linux and mac.

## Features

- Detect objects in saved images, clipboard images, webcam, screen, or video files
- Save annotated videos to disk
- Per-instance tracking and numbering across video frames
- Adaptive label coloring (black/white) based on background brightness
- Frame-skipping for lower CPU load on live sources
- Change-based terminal logging — only prints when detected objects change

## Installation

```bash
git clone https://github.com/CarlosEckert/object-detection-core.git
cd object-detection-core
pip install -r requirements.txt
```

The YOLO model weights (`yolov8m-oiv7.pt`) are downloaded automatically by Ultralytics on first run and placed in `models/`.

## Usage

Edit `main.py` to pick a detection mode, then run:

```bash
python main.py
```

Available entry points (in `src/detection_runners.py`):

| Function | Purpose |
| --- | --- |
| `detect_saved_image(path)` | Run detection on an image file |
| `detect_clipboard_image()` | Detect objects in the current clipboard image |
| `detect_camera(index=0)` | Live webcam detection |
| `detect_screen()` | Live screen-capture detection |
| `detect_saved_video_live(path)` | Play a video file with live detection overlay |
| `detect_saved_video_then_saveit(in, out)` | Run detection on a video and save the annotated result |

#

## Configuration

Tune these in `main.py` before calling a runner:

| Setting | Default | Description |
| --- | --- | --- |
| `detection_core.CONF` | `0.15` | Minimum confidence for a detection to be considered |
| `detection_core.LABELING_CONF` | `0.20` | Minimum confidence for the on-screen text label |
| `detection_core.IOU` | `0.45` | IoU threshold for deduplication (lower = more aggressive) |
| `detection_core.SHOW_CONF_IN_PREVIEW` | `True` | Render confidence values inside the preview window |
| `detection_runners.REMOVE_DOWNSCALING_IN_VIDEOS` | `False` | Disable YOLO's 640-px downscale (~9× compute at 1080p, better small-object recall) |
| `detection_runners.ANALYSE_EVERY_X_FRAME` | `3` | Run YOLO every Nth frame; skipped frames redraw the cached boxes |

### Hardware acceleration

Inference runs on the **GPU only if you have an NVIDIA GPU** with a CUDA-enabled build of PyTorch installed. 
On all other systems (AMD, Intel, integrated graphics, or CPU-only) it falls back to the **CPU**. 
Ultralytics auto-selects the device — you don't need to configure anything.

> ⚠️ **Temperature warning:** sustained inference on video files or live video pushes the GPU or CPU hard.   
> Watch your component temps, especially the cpu temp and if you use a laptop, and even more if you want to set `REMOVE_DOWNSCALING_IN_VIDEOS = True` and you run on cpu.

### Recommended performance settings (rough estimates only, depends on your hardware)

| Setting | NVIDIA GPU | CPU only |
| --- | --- | --- |
| `REMOVE_DOWNSCALING_IN_VIDEOS` | `True`  | `False` |
| `ANALYSE_EVERY_X_FRAME` | `1` (every frame) | `3`–`6`  |

For live video, raise the value of ANALYSE_EVERY_X_FRAME until the preview runs at the same speed as the normal video.       
For saved images and `detect_clipboard_image` these settings don't matter — they always run a single inference at native resolution.

#

## Project Structure

```
.
├── main.py                    # entry point; pick a runner and tune settings
├── src/
│   ├── detection_core.py      # YOLO inference, drawing, tracking, terminal output
│   └── detection_runners.py   # high-level runners (image / video / camera / screen)
├── models/                    # YOLO weights (downloaded on first run)
└── requirements.txt
```

`detection_core` and `detection_runners` are designed to be imported by downstream projects. Build on top of them instead of forking — keep the inference layer here and add new behavior in your own module.

## Requirements

- Python 3.12+
- See `requirements.txt` for package versions

## Notes

- **Windows-only quirk:** screen-capture mode marks the preview window as invisible to capture APIs so it doesn't show up in its own feed.    
This may also work on other operating systems, you can solve problems there by moving the preview window to a secondary display
- **Clipboard image not detected?** (Enable clipboard history and) click the image once in the clipboard manager before running.
