import time
import cv2
from detection_core import init_yolo, run_yolo_tracker


def detect_anomalies(video_source=0, scan_duration=5, percentage=80):
    """
    video_source: 0 (or other int) for camera, or a path to a video file.
    scan_duration: seconds to scan the scene to learn what's normal.
    percentage: a class must appear in at least this % of scan frames to be baseline.
    """
    print(f"Scanning for {scan_duration}s to build baseline...")
    model = init_yolo()
    id_map = {}
    frame_class_counts = []  # one dict per scan frame: {label: count}
    baseline = None  # set after scan: {label: max_count}
    reported_anomalies = set()
    start_time = time.time()

    if isinstance(video_source, int):
        source = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
    else:
        source = cv2.VideoCapture(video_source)

    win_name = 'Anomaly Detection'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while cv2.waitKey(1) != 27:  # Escape
        has_frame, frame = source.read()
        if not has_frame:
            break

        counts = run_yolo_tracker(model, frame, False, True, id_map)
        elapsed = time.time() - start_time

        if baseline is None:
            # Scan phase: collect per-frame counts
            frame_class_counts.append(counts)
            if elapsed >= scan_duration:
                total_frames = len(frame_class_counts)
                min_frames = total_frames * percentage / 100
                # how many frames each class appeared in
                class_appearance = {}
                for fc in frame_class_counts:
                    for cls in fc:
                        class_appearance[cls] = class_appearance.get(cls, 0) + 1
                # keep only classes seen in >= percentage% of frames; baseline count = max instances seen
                baseline = {}
                for cls, app in class_appearance.items():
                    if app >= min_frames:
                        baseline[cls] = max(fc.get(cls, 0) for fc in frame_class_counts)
                print(f"Scan complete. Baseline: {baseline}")
        else:
            # Detection phase: flag classes/instances not in baseline
            for cls, cnt in counts.items():
                if cls not in baseline:
                    if ('new', cls) not in reported_anomalies:
                        print(f"ANOMALY: new object class detected: {cls}")
                        reported_anomalies.add(('new', cls))
                elif cnt > baseline[cls]:
                    key = ('extra', cls, cnt)
                    if key not in reported_anomalies:
                        print(f"ANOMALY: extra {cls} detected ({cnt} vs baseline {baseline[cls]})")
                        reported_anomalies.add(key)

        cv2.imshow(win_name, frame)

    source.release()
    cv2.destroyWindow(win_name)
