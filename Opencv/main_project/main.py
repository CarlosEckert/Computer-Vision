import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mss
import keyboard


import sys
import pyautogui


def write_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)     # neues bild, 1dim = höhe, 2dim = breite, 3dim = Farbkanäle (3=bgr)
    image[:] = (255, 255, 0)  # Alle Pixel bekommen blaue farbe
    cv2.imwrite("image.jpeg", image)


def read_image():
    img = cv2.imread("image.jpeg", 1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)     # zeichnet sozusagen das bild
    plt.show()          # wird benötigt um alle plt bilder anzuzeigen


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


def video_from_camera():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1920)  # Breite
    cap.set(4, 1080)  # Höhe
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20
    print(frame_width, frame_height)
    out_mp4 = cv2.VideoWriter("camera_recording.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (frame_width, frame_height))

    print("Starting recording camera, Press ESC to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out_mp4.write(frame)
            preview = cv2.resize(frame, (640, 360))
            cv2.imshow("Recording", preview)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out_mp4.release()
    cv2.destroyAllWindows()


def video_from_screen():
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # 1 = Hauptbildschirm
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 20
        out = cv2.VideoWriter("screen_recording.avi", fourcc, fps, (monitor["width"], monitor["height"]))

        print("Starting recording screen, Press ESC to quit.")

        while True:
            frame = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            out.write(frame)

            if keyboard.is_pressed("esc"):
                break

    out.release()



video_from_camera()