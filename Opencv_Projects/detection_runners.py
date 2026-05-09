import cv2
import ctypes
import time
import numpy as np
from PIL import ImageGrab
from detection_core import init_yolo, analyse_frame, redraw_detections


REMOVE_DOWNSCALING_IN_VIDEOS = False
ANALYSE_EVERY_X_FRAME = 1
WARMUP_FRAMES = 5  # for video sources, suppress change-detected output for the first N frames while YOLO settles


def process_image(image_path=None, frame=None, remove_downscaling=True):
    print("Press ESC to quit.")
    model = init_yolo()

    if frame is None and image_path is not None:
        frame = cv2.imread(image_path)

    elif frame is not None and image_path is None:
        frame = frame  # just to clarify

    else:
        print("Detect_image must be called with image path or frame, must be set as the right keyword argument")

    analyse_frame(model, frame, track_bool=False, remove_downscaling=remove_downscaling)

    win_name = 'Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def detect_saved_image(image_path):
    process_image(image_path)


# if you have a problem with images not being detected that you copied some time ago, (enable clipboard history and) from the clipboard manager click the image once that you want to analyze before running the code
def detect_clipboard_image():
    clipboard_image = ImageGrab.grabclipboard()
    if clipboard_image is None:
        print("No image found in clipboard.")
        return
    frame = cv2.cvtColor(np.array(clipboard_image), cv2.COLOR_RGB2BGR)
    process_image(frame=frame)



def process_video(source, track=True, output_path=None):
    save_mode = output_path is not None

    if save_mode:
        print(f"Saving annotated video to {output_path}...")
    else:
        print("Press ESC to quit.")

    model = init_yolo()
    frame_count = 0
    analysed_count = 0   # counts only frames that actually went through YOLO; used for warmup gating
    id_map = {}
    last_drawables = []  # cached detections to redraw on skipped frames

    cap, win_name, is_screen = _open_source(source)

    # Saved videos in preview mode need pacing — without it, fast loops (e.g. ANALYSE_EVERY_X_FRAME > 1)
    # blast through the file. Camera/screen don't need this (live sources have their own clock); save mode
    # doesn't want this (we write as fast as possible).
    needs_pacing = not is_screen and not isinstance(source, int) and not save_mode
    pacing_start_time = time.time() if needs_pacing else None

    writer = None
    fps = 0
    total_frames = 0
    analysis_start = None
    if save_mode:
        writer, fps, total_frames = _open_writer(cap, output_path)
        analysis_start = time.time()

    # Set up a resizable preview window (skipped in save mode since there's no preview).
    # For screen capture, default the initial size to 1/3 of the screen resolution so 4K and FHD monitors look comparable.
    if not save_mode:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if is_screen:
            screen_width, screen_height = ImageGrab.grab().size
            cv2.resizeWindow(win_name, screen_width // 3, screen_height // 3)
            # Force the window to actually be created at the OS level (namedWindow alone doesn't),
            # then mark it excluded from screen capture so it disappears from ImageGrab's view.
            cv2.imshow(win_name, np.zeros((100, 100, 3), dtype=np.uint8))
            cv2.waitKey(1)
            _exclude_window_from_capture(win_name)

    while True:
        # In preview mode the waitKey is needed to pump window events and check ESC; in save mode we skip it for speed
        if not save_mode and cv2.waitKey(1) == 27:
            break

        if is_screen:
            screenshot = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            has_frame, frame = cap.read()
            if not has_frame:
                break

        # analyse_every_x_frame <= 1 disables the skip logic (analyse every frame); otherwise only analyse every Xth frame
        should_analyse = ANALYSE_EVERY_X_FRAME <= 1 or frame_count % ANALYSE_EVERY_X_FRAME == 0
        if should_analyse:
            silent = analysed_count < WARMUP_FRAMES
            _, last_drawables = analyse_frame(model, frame, track_bool=track, id_map=id_map, remove_downscaling=REMOVE_DOWNSCALING_IN_VIDEOS, silent=silent)
            analysed_count += 1
        else:
            # Skipped frame — redraw the last known boxes so they don't flicker on/off
            redraw_detections(frame, last_drawables)
        frame_count += 1

        if save_mode:
            writer.write(frame)
        else:
            cv2.imshow(win_name, frame)

        if needs_pacing:
            _pace_to_video_clock(cap, pacing_start_time)

    if cap is not None:
        cap.release()
    if save_mode:
        writer.release()
        _print_save_summary(output_path, fps, total_frames, analysis_start)
    else:
        cv2.destroyWindow(win_name)



def _open_source(source):
    """Open the appropriate capture source. Returns (cap, win_name, is_screen)."""
    if source == 'screen':
        return None, 'Screen Detection', True
    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW), 'Camera Preview', False
    return cv2.VideoCapture(source), 'Video Detection', False


def _open_writer(cap, output_path):
    """Set up an mp4 writer matching the source's resolution and FPS. Returns (writer, fps, total_frames)."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height)), fps, total_frames


def _pace_to_video_clock(cap, pacing_start_time):
    """Sleep so wall-clock time matches the next frame's timestamp in the source video."""
    next_frame_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    elapsed_ms = (time.time() - pacing_start_time) * 1000
    sleep_for = (next_frame_ms - elapsed_ms) / 1000
    if sleep_for > 0:
        time.sleep(sleep_for)


def _print_save_summary(output_path, fps, total_frames, analysis_start):
    """Print the post-save block (output path, source duration, analysis time)."""
    elapsed = time.time() - analysis_start
    video_duration = total_frames / fps
    print(f"\n\nDone. Saved analysed video to {output_path}")
    print(f"  Source video: {video_duration} sec at {fps:.2f} FPS")
    print(f"  Analysis took: {elapsed:.2f} sec")


def _exclude_window_from_capture(win_name):
    """Mark a created OpenCV window as invisible to screen-capture APIs (Windows-only; harmless elsewhere)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, win_name)
        if hwnd:
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except (AttributeError, OSError):
        pass  # not Windows or unsupported version




def detect_camera(camera_index=0):
    process_video(camera_index)


def detect_screen(track=False):
    process_video('screen', track=track)


def detect_saved_video_live(video_path):
    process_video(video_path)


def detect_saved_video_then_saveit(video_path, output_path):
    process_video(video_path, output_path=output_path)


