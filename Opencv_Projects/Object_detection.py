import cv2
from ultralytics import YOLO



# Braucht Rechte
def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=2)


def init_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


def run_yolo_tracker(model, frame, print_bool, track_bool, id_map=None):
    if track_bool:
        results = model.track(frame, persist=True, verbose=print_bool)
    else:
        results = model.predict(frame, verbose=print_bool)

    if results[0].boxes is not None:
        class_counts = {}
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)

            label = results[0].names[int(box.cls[0])]
            if track_bool and box.id is not None and id_map is not None:
                key = (label, int(box.id[0]))
                if key not in id_map:
                    class_num = sum(1 for k in id_map if k[0] == label) + 1
                    id_map[key] = class_num
                number = id_map[key]
            else:
                class_counts[label] = class_counts.get(label, 0) + 1
                number = class_counts[label]

            cv2.putText(frame, f'{label} {number}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)



def detect_camera():
    print("Starting usage of camera, Press ESC to quit.")
    model = init_yolo()
    frame_count = 0
    print_interval = 60  # does not consern the tracking and is just the interval for terminal prints which act as a save point / history
    id_map = {}

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

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

    cv2.imshow('Detection', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('Detection')


def detect_video(video_path):
    print("Press ESC to quit.")
    model = init_yolo()
    id_map = {}

    source = cv2.VideoCapture(video_path)
    frame_count = 0
    print_interval = 60

    win_name = 'Video Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

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