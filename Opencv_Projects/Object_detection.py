import cv2
from ultralytics import YOLO



# Braucht Rechte
def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=2)


def init_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


def run_yolo_tracker(model, frame, print_bool):
    results = model.track(frame, persist=True, verbose=print_bool)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)
            label = results[0].names[int(box.cls[0])]
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)    # = text aboth the object rectangle


def use_camera():
    print("Starting usage of camera, Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60  # does not consern the tracking and is just the interval for terminal prints which act as a save point / history

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        print_bool = frame_count % print_interval == 0
        run_yolo_tracker(model, frame, print_bool)
        frame_count += 1
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)


use_camera()