import detection_core
import detection_runners
from detection_runners import (
    detect_saved_image, detect_clipboard_image,
    detect_camera, detect_saved_video_live, detect_saved_video_then_saveit,
    detect_screen,
)
from anomaly_detector import detect_anomalies


if __name__ == '__main__':

    # YOLO detection thresholds. Lower values = more detections but more false positives
    detection_core.CONF = 0.15            #  minimum confidence for a detection to be considered at all
    detection_core.LABELING_CONF = 0.20   # minimum confidence for the on-screen text label, also any object between CONF and LABELING_CONF will be printed as a low_conf object
    detection_core.IOU = 0.45             # lower = more aggressive deduplication


    # False: Will print the confidence of each object in the terminal but will not show in preview. True: Will also show the conf in preview
    detection_core.SHOW_CONF_IN_PREVIEW = True

    # True: will remove the standard downscaling: Will increase the computational intensity (9x for full hd) but will help detecting small objects, be wary of temperatures
    detection_runners.REMOVE_DOWNSCALING_IN_VIDEOS = False

    # if you want to have less computational load or the preview or processing of detecting a saved video is to slow, increase this number
    detection_runners.ANALYSE_EVERY_X_FRAME = 3     # if you use detect_saved_video_live and the video looks slower than usual, increase this number right until the video has normal speed


    #detect_saved_image('chair_image.png')
    #detect_clipboard_image()

    #detect_camera()
    #detect_screen()

    detect_saved_video_live(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4')
    #detect_saved_video_then_saveit(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4', r'C:\Users\carlo\Videos\SteelSeries Moments\detection_test.mp4')

    #detect_anomalies(0, scan_duration=5, percentage=80)
