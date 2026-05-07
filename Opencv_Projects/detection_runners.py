import cv2
import numpy as np
from PIL import ImageGrab
from detection_core import init_yolo, run_yolo_tracker


def process_image(image_path=None, frame=None, remove_downscaling=True):
    print("Press ESC to quit.")
    model = init_yolo()

    if frame is None and image_path is not None:
        frame = cv2.imread(image_path)

    elif frame is not None and image_path is None:
        frame = frame  # just to clarify

    else:
        print("Detect_image must be called with image path or frame, must be set as the right keyword argument")

    run_yolo_tracker(model, frame, True, False, remove_downscaling=remove_downscaling)

    win_name = 'Detection'
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






# here the removal of the downscaling should be done carefully. Which one frame the computational intensity inst a problem, in a video it is a different story
# CAREFUL if you change the remove_downscaling to True here, the task manager might not show differences in cpu utilization, but the cpu temperature will go up rapidly (in my case 65° celsius vs 83° celsius)
# should only be a problem if you don't have a nvidea gpu but no guaranties (for context I have an amd gpu so no Cuda so everything runs on the cpu)
def process_video(source, track=True, remove_downscaling=False):
    print("Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60  # does not concern the tracking and is just the interval for terminal prints which act as a save point / history
    id_map = {}

    is_screen = source == 'screen'
    if is_screen:
        cap = None
        win_name = 'Screen Detection'
    elif isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        win_name = 'Camera Preview'
    else:
        cap = cv2.VideoCapture(source)
        win_name = 'Video Detection'

    while cv2.waitKey(1) != 27:  # Escape
        if is_screen:
            screenshot = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            has_frame, frame = cap.read()
            if not has_frame:
                break

        print_bool = frame_count % print_interval == 0
        run_yolo_tracker(model, frame, print_bool, track, id_map, remove_downscaling=remove_downscaling)
        frame_count += 1
        cv2.imshow(win_name, frame)

    if cap is not None:
        cap.release()
    cv2.destroyWindow(win_name)


def detect_camera(camera_index=0):
    process_video(camera_index)


def detect_saved_video(video_path):
    process_video(video_path)


def detect_screen():
    process_video('screen')




