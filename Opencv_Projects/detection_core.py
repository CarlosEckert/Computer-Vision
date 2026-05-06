import cv2
from ultralytics import YOLO


DETECT_EVERYTHING = False


def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=1)


# oiv7 modell is trained on the open images 7 databank with 600 object classes
def init_yolo(model_path='yolov8m-oiv7.pt'):
    return YOLO(model_path)


# conf = how sure is the tracker that the object is even an object            iou=percentage of overlapping between two objects that is allowed for it to be classified as two
# pixel_width_after_compressing: by stopping the model from compressing the image, the detection works better for small objects at the cost of computational intensity. 640 is baseline, 1920 would be 9x the computation
def run_yolo_tracker(model, frame, print_bool, track_bool, id_map=None, conf=0.1, labeling_conf=0.3, iou=0.45, max_det=300, pixel_width_after_compressing=640):
    if DETECT_EVERYTHING:
        conf = 0.01
        labeling_conf = 0.01
        iou = 0.7


    if track_bool:
        results = model.track(frame, persist=True, verbose=print_bool, conf=conf, iou=iou, max_det=max_det, imgsz=pixel_width_after_compressing)
    else:
        results = model.predict(frame, verbose=print_bool, conf=conf, iou=iou, max_det=max_det, imgsz=pixel_width_after_compressing)

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
        low_conf_items = []  # (label, confidence) for instances below the 2x conf threshold
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)

            label = results[0].names[int(box.cls[0])]
            box_conf = float(box.conf[0])
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

            label_text = f'{label} {number} {box_conf:.2f}'

            # When DETECT_EVERYTHING is False, only show the classification text on the preview and in the terminal if the instance confidence reaches label_conf.
            # Low-confidence detections still get a rectangle, just no label.
            show_label = box_conf >= labeling_conf

            if show_label:
                # If the box is too close to the top of the frame to fit the label above it, draw the label below the rectangle instead.
                if y1 < 10:
                    text_y = y2 + 10
                else:
                    text_y = y1 - 8
                cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if print_bool:
                if show_label:
                    print(f'  {label_text}')
                else:
                    low_conf_items.append((label, box_conf))

        if print_bool and low_conf_items:
            items_str = ', '.join(f'{lbl} {c:.2f}' for lbl, c in low_conf_items)
            print(f'  + {len(low_conf_items)} low-confidence detection(s): {items_str}')
    return frame_counts
