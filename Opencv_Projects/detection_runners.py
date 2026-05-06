import cv2
import numpy as np
from PIL import ImageGrab
from detection_core import init_yolo, run_yolo_tracker


def detect_camera():
    print("Starting usage of camera, Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60  # does not consern the tracking and is just the interval for terminal prints which act as a save point / history
    id_map = {}

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    win_name = 'Camera Preview'

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        print_bool = frame_count % print_interval == 0
        run_yolo_tracker(model, frame, print_bool, True, id_map)
        frame_count += 1
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)


def detect_image(image_path=None, frame=None):
    print("Press ESC to quit.")
    model = init_yolo()

    if frame is None and image_path is not None:
        frame = cv2.imread(image_path)

    elif frame is not None and image_path is None:
        frame = frame  # just to clarify

    else:
        print("Detect_image must be called with image path or frame, must be set as the right keyword argument")

    run_yolo_tracker(model, frame, True, False)

    win_name = 'Detection'
    cv2.imshow(win_name, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def detect_clipboard_image():
    clipboard_image = ImageGrab.grabclipboard()
    if clipboard_image is None:
        print("No image found in clipboard.")
        return
    frame = cv2.cvtColor(np.array(clipboard_image), cv2.COLOR_RGB2BGR)
    detect_image(frame=frame)



def detect_video(video_path):
    print("Press ESC to quit.")
    model = init_yolo()
    id_map = {}

    source = cv2.VideoCapture(video_path)
    frame_count = 0
    print_interval = 60

    win_name = 'Video Detection'

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        print_bool = frame_count % print_interval == 0
        run_yolo_tracker(model, frame, print_bool, True, id_map)
        frame_count += 1
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)


# works best with a second monitor that has the same resolution, in that case delete the resizing and push the detection window to the second monitor
def detect_screen():
    print("Starting screen detection. Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60
    id_map = {}

    win_name = 'Screen Detection'

    while cv2.waitKey(1) != 27:  # Escape
        screenshot = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        #frame = cv2.resize(frame, (883, 480))

        print_bool = frame_count % print_interval == 0
        run_yolo_tracker(model, frame, print_bool, False, id_map)
        frame_count += 1

        cv2.imshow(win_name, frame)

    cv2.destroyWindow(win_name)




#detect_camera()
#detect_image(image_path='chair2.jpg')
#detect_clipboard_image()
#detect_video(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4')
detect_screen()
