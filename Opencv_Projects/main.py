from detection_runners import detect_camera, detect_saved_image, detect_clipboard_image, detect_saved_video, detect_screen
import detection_core
from anomaly_detector import detect_anomalies


if __name__ == '__main__':
    # False: The Detection will be reasonable, True: Nearly everything that the model thinks could be an objects will be marked as one
    detection_core.DETECT_EVERYTHING = False

    #detect_saved_image(r'C:\downloads\chair tesst.jpg')
    detect_clipboard_image()

    # if the detection of small objects doesn't work well, go to the process_video function header and change the remove_downscaling to true. Mind the warning on top in regards to computational complexity
    #detect_camera()
    #detect_saved_video(r'')
    #detect_screen()

    #detect_anomalies(0, scan_duration=5, percentage=80)
