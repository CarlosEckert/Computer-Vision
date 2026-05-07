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
def run_yolo_tracker(model, frame, print_bool, track_bool, id_map=None, conf=0.2, labeling_conf=0.2, iou=0.45, max_det=300, pixel_width_after_compressing=640):
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
        low_conf_items = []
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

            draw_label(frame, x1, y1, label_text, show_label)

            if print_bool:
                if show_label:
                    print(f'{label_text}')
                else:
                    low_conf_items.append((label, box_conf))

        if print_bool and low_conf_items:
            items_str = ', '.join(f'{lbl} {c:.2f}' for lbl, c in low_conf_items)
            print(f'  + {len(low_conf_items)} low-confidence detection(s): {items_str}')
    return frame_counts



def draw_label(frame, x1, y1, label_text, show_label):
    if not show_label:
        return

    label_text = label_text[:-4]        # will delete the confidence text from the preview. You can see it in the terminal but this can also be removed to show the confidence in the frame

    frame_height, frame_width = frame.shape[:2]

    if y1 < 14:                          # text above the top line would be out of frame, put it under the top line instead
        text_y = y1 + 15
        x1 = x1 + 2
    else:                                # normal case, text sits on top of the top line
        text_y = y1 - 5


    # Compute the three sample coordinates: left start, horizontal middle, right end of where the text will land, all on the text's vertical centerline.
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_middle_y = text_y - text_h // 2            # this is just the y coordinate of the middle of the text, text_y is the bottom
    start_coord = (x1, text_middle_y)
    middle_coord = (x1 + text_w // 2, text_middle_y)
    end_coord = (x1 + text_w - 1, text_middle_y)


    # Default text color is black; switch to white if at least 2 of the 3 sample points are dark.
    text_color = (0, 0, 0)
    dark_count = 0
    for (sample_x, sample_y) in (start_coord, middle_coord, end_coord):
        if 0 <= sample_x < frame_width and 0 <= sample_y < frame_height:
            b, g, r = frame[sample_y, sample_x]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            if brightness < 50:
                dark_count += 1
    if dark_count >= 3:
        text_color = (255, 255, 255)

    cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
