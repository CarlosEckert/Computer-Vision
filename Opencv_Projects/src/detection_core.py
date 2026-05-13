import cv2
from ultralytics import YOLO


SHOW_CONF_IN_PREVIEW = False

CONF = 0.15
LABELING_CONF = 0.20
IOU = 0.45

_last_all_keys = None


def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=1)


# oiv7 modell is trained on the open images 7 databank with 600 object classes
def init_yolo(model_path='models/yolov8m-oiv7.pt'):
    return YOLO(model_path)



def analyse_frame(model, frame, track_bool, id_map=None, max_det=300, remove_downscaling=False, silent=False):
    results = _run_inference(model, frame, track_bool, max_det, remove_downscaling)
    boxes = results[0].boxes
    names = results[0].names

    if track_bool and id_map is not None:
        _release_unused_instance_numbers(id_map, boxes, names)

    frame_counts = {}
    drawables = []  # list of (x1, y1, x2, y2, label_text, show_label) tuples for re-drawing on skipped frames
    high_conf_items = []  # collected for change-based terminal output
    low_conf_items = []
    if boxes is not None:
        class_counts = {}
        for box in boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)

            label = names[int(box.cls[0])]
            box_conf = float(box.conf[0])
            frame_counts[label] = frame_counts.get(label, 0) + 1
            number = _assign_instance_number(label, box, track_bool, id_map, class_counts)

            label_text = f'{label} {number}: {box_conf:.2f}'

            # Only show the classification text on the preview and in the terminal if the instance
            # confidence reaches LABELING_CONF. Low-confidence detections still get a rectangle.
            show_label = box_conf >= LABELING_CONF

            draw_label(frame, x1, y1, label_text, show_label)
            drawables.append((x1, y1, x2, y2, label_text, show_label))

            if show_label:
                high_conf_items.append((label, number, box_conf))
            else:
                low_conf_items.append((label, number, box_conf))

    print_detections(high_conf_items, low_conf_items, silent=silent)
    return frame_counts, drawables



def _run_inference(model, frame, track_bool, max_det, remove_downscaling):
    # Round the frame's width up to the next multiple of 32 (YOLO's stride requirement).
    # remove_downscaling=True keeps the frame's native resolution (better for small objects, ~9x compute at 1920p).
    if remove_downscaling:
        imgsz = ((frame.shape[1] + 31) // 32) * 32
    else:
        imgsz = 640

    if track_bool:
        return model.track(frame, persist=True, verbose=False, conf=CONF, iou=IOU, max_det=max_det, imgsz=imgsz)
    else:
        return model.predict(frame, verbose=False, conf=CONF, iou=IOU, max_det=max_det, imgsz=imgsz)



def _release_unused_instance_numbers(id_map, boxes, names):
    visible_keys = set()
    if boxes is not None:
        for box in boxes:
            if box.id is not None:
                label = names[int(box.cls[0])]
                visible_keys.add((label, int(box.id[0])))
    for k in list(id_map.keys()):
        if k not in visible_keys:
            del id_map[k]



def _assign_instance_number(label, box, track_bool, id_map, class_counts):
    if track_bool and box.id is not None and id_map is not None:
        key = (label, int(box.id[0]))
        if key not in id_map:
            used = {v for k, v in id_map.items() if k[0] == label}
            num = 1
            while num in used:
                num += 1
            id_map[key] = num
        return id_map[key]
    class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts[label]




# Re-draw cached detections on a fresh frame without re-running YOLO. Used when ANALYSE_EVERY_X_FRAME > 1
# so skipped frames show the last known boxes instead of flickering between annotated and raw.
def redraw_detections(frame, drawables):
    for x1, y1, x2, y2, label_text, show_label in drawables:
        draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)
        draw_label(frame, x1, y1, label_text, show_label)



def draw_label(frame, x1, y1, label_text, show_label):
    if not show_label:
        return

    if not SHOW_CONF_IN_PREVIEW:
        label_text = label_text[:-6]

    frame_height, frame_width = frame.shape[:2]

    if y1 < 14:                          # text above the top line would be out of frame, put it under the top line instead
        text_y = y1 + 15
        x1 = x1 + 2
    else:                                # normal case, text sits on top of the top line
        text_y = y1 - 5


    # Compute the three sample coordinates: left start, horizontal middle, right end of where the text will land, all on the text's vertical centerline.
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_middle_y = text_y - text_h // 2            # this is just the y coordinate of the middle of the text, text_y is the bottom

    start_coord = (x1, text_middle_y)
    middle_coord = (x1 + text_w // 2, text_middle_y)
    end_coord = (x1 + text_w - 1, text_middle_y)
    start_to_middle_coord = ((start_coord[0] + middle_coord[0]) // 2, text_middle_y)
    middle_to_end_coord = ((middle_coord[0] + end_coord[0]) // 2, text_middle_y)


    # Default text is black, change to white if to much of the background is dark
    text_color = (0, 0, 0)
    dark_count = 0
    for (sample_x, sample_y) in (start_coord, middle_coord, end_coord, start_to_middle_coord, middle_to_end_coord):
        if 0 <= sample_x < frame_width and 0 <= sample_y < frame_height:
            b, g, r = frame[sample_y, sample_x]
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            if 60 > brightness > 1:     # >1 for a special case where the background is a drawn rectangle line
                dark_count += 1
    if dark_count >= 2:
        text_color = (255, 255, 255)

    cv2.putText(frame, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)



def print_detections(high_conf_items, low_conf_items, silent=False):
    # During warmup (first few video frames where YOLO is still settling) the runner asks
    # us to stay quiet. We also skip the state update, so the first non-silent call is
    # treated as the true "first call" and starts with a clean slate.
    if silent:
        return

    global _last_all_keys

    # Build a single union of (label, number) identifiers across both categories. Comparing
    # this union ignores high↔low transitions (the identifier stays in the union either way)
    # and only triggers on genuine appearances and disappearances.
    all_keys = frozenset((label, num) for label, num, _ in high_conf_items) | \
               frozenset((label, num) for label, num, _ in low_conf_items)

    if all_keys == _last_all_keys:
        return

    # The very first call has no previous state to "change" from, so use a different header.
    is_first_call = _last_all_keys is None

    _last_all_keys = all_keys

    if is_first_call:
        print("\n\n detected objects:")
    else:
        print("\n\nchange detected, objects:")

    # All objects disappeared (nothing high, nothing low) — special-case message.
    if not high_conf_items and not low_conf_items:
        print("    no objects detected")
        return

    for label, number, box_conf in high_conf_items:
        print(f'    {label} {number}: {box_conf:.2f}')
    if low_conf_items:
        items_str = ', '.join(f'{lbl} {c:.2f}' for lbl, _, c in low_conf_items)
        print(f'        {len(low_conf_items)} low-confidence detection(s): {items_str}')
