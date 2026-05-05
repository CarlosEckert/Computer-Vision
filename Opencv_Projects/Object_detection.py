import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mss
import keyboard
from ultralytics import YOLO



# Braucht Rechte
def draw_rectangle(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), thickness=2)


def init_yolo(model_path='yolov8n.pt'):
    return YOLO(model_path)


def run_yolo_tracker(model, frame):
    results = model.track(frame, persist=True)
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            draw_rectangle(frame, x1, y1, x2 - x1, y2 - y1)


def use_camera():
    print("Starting usage of camera, Press ESC to quit.")
    model = init_yolo()

    source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        run_yolo_tracker(model, frame)
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)






use_camera()