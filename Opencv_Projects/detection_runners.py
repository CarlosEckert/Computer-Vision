import cv2
from detection_core import init_yolo, run_yolo_tracker, WINDOW_WIDTH, WINDOW_HEIGHT


def detect_camera():
    print("Starting usage of camera, Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60  # does not consern the tracking and is just the interval for terminal prints which act as a save point / history
    id_map = {}

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_WIDTH, WINDOW_HEIGHT)

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


def detect_image(image_path):
    print("Press ESC to quit.")
    model = init_yolo()

    frame = cv2.imread(image_path)
    run_yolo_tracker(model, frame, True, False)

    win_name = 'Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.imshow(win_name, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)


def detect_video(video_path):
    print("Press ESC to quit.")
    model = init_yolo()
    id_map = {}

    source = cv2.VideoCapture(video_path)
    frame_count = 0
    print_interval = 60

    win_name = 'Video Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, WINDOW_WIDTH, WINDOW_HEIGHT)

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


detect_camera()
# detect_image('path/to/image')
# detect_video('path/to/video')
