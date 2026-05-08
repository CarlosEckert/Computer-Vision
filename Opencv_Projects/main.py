import detection_core
import detection_runners
from detection_runners import (
    detect_saved_image, detect_clipboard_image,
    detect_camera, detect_saved_video_live, detect_saved_video_then_saveit,
    detect_screen,
)
from anomaly_detector import detect_anomalies


if __name__ == '__main__':
    # False: The Detection will be reasonable. True: Nearly everything that the model thinks could be an objects will be marked as one
    detection_core.DETECT_EVERYTHING = False

    # False: Will print the confidence of each object in the terminal but will not show in preview. True: Will also show the conf in preview
    detection_core.SHOW_CONF_IN_PREVIEW = True

    # True: will remove the standard downscaling: Will increase the computational intensity (9x for full hd) but will help detecting small objects, be wary of temperatures
    detection_runners.REMOVE_DOWNSCALING_IN_VIDEOS = False

    # if you want to have less computational load or the preview or processing of detecting a saved video is to slow, increase this number
    detection_runners.ANALYSE_EVERY_X_FRAME = 10     # 3 only looks normal because now the inference time = typical frame time. the inference isnt beeing addet to the frame time because analysing the frame is quicker then the frame time


    #detect_saved_image('chair_image.png')
    #detect_clipboard_image()

    #detect_camera()
    #detect_saved_video_live(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4')
    detect_saved_video_then_saveit(r'C:\Users\carlo\Videos\SteelSeries Moments\Counter-Strike-2__2026-04-22__22-21-03.mp4', r'C:\Users\carlo\Videos\SteelSeries Moments\detection_test.mp4')
    #detect_screen()

    #detect_anomalies(0, scan_duration=5, percentage=80)
