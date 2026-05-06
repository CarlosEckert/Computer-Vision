from detection_runners import detect_camera, detect_saved_image, detect_clipboard_image, detect_saved_video, detect_screen
import detection_core
from anomaly_detector import detect_anomalies


if __name__ == '__main__':
    # False: The Detection will be reasonable, True: Nearly everything that the model thinks could be an objects will be marked as one
    detection_core.DETECT_EVERYTHING = False

    # detect_camera()
    # detect_saved_image(image_path='chair2.jpg')
    # detect_clipboard_image()
    # detect_saved_video(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4')
    detect_screen()

    #detect_anomalies(0, scan_duration=5, percentage=80)
