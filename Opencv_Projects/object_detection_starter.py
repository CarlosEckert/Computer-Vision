from detection_runners import detect_camera, detect_image, detect_video
from anomaly_detector import detect_anomalies


if __name__ == '__main__':
    detect_camera()
    # detect_image('path/to/image.jpg')
    # detect_video('path/to/video.mp4')
    # detect_anomalies(0, scan_duration=5, percentage=80)
