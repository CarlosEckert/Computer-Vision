import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mss
import keyboard


# Braucht Rechte
def use_camera():
    print("Starting usage of camera, Press ESC to quit.")
    source = cv2.VideoCapture(0, cv2.CAP_MSMF)

    win_name = 'Camera Preview'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)


def draw_rectangle(frame, x, y, width, height):
    color = (0, 0, 0)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness=2)
    cv2.imshow('Rectangle', frame)
    cv2.waitKey(0)
    cv2.destroyWindow('Rectangle')






use_camera()