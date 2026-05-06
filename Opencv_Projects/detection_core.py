import cv2
from ultralytics import YOLO


WINDOW_WIDTH = 480
WINDOW_HEIGHT = 270


def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=2)


def init_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


def run_yolo_tracker(model, frame, print_bool, track_bool, id_map=None):
    if track_bool:
        results = model.track(frame, persist=True, verbose=print_bool)
    else:
        results = model.predict(frame, verbose=print_bool)

    boxes = results[0].boxes

    # Prune stale tracker IDs from id_map (those no longer visible this frame)
    if track_bool and id_map is not None:
        visible_keys = set()
        if boxes is not None:
            for box in boxes:
                if box.id is not None:
                    label = results[0].names[int(box.cls[0])]
                    visible_keys.add((label, int(box.id[0])))
        for k in list(id_map.keys()):
            if k not in visible_keys:
                del id_map[k]

    frame_counts = {}
    if boxes is not None:
        class_counts = {}
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)

            label = results[0].names[int(box.cls[0])]
            frame_counts[label] = frame_counts.get(label, 0) + 1

            if track_bool and box.id is not None and id_map is not None:
                key = (label, int(box.id[0]))
                if key not in id_map:
                    # assign lowest unused number for this class
                    used = {v for k, v in id_map.items() if k[0] == label}
                    num = 1
                    while num in used:
                        num += 1
                    id_map[key] = num
                number = id_map[key]
            else:
                class_counts[label] = class_counts.get(label, 0) + 1
                number = class_counts[label]

            cv2.putText(frame, f'{label} {number}', (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame_counts
