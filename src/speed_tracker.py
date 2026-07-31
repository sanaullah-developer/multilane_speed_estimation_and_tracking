"""
Multi-Lane Vehicle Speed Estimation
------------------------------------
Pipeline: YOLO detection -> ByteTrack tracking -> homography-based
pixel->meters conversion -> smoothed speed estimate -> lane assignment
-> violation logging.

Run calibrate.py FIRST on a sample frame to generate calibration.json
(homography source/dest points + lane polygons) before running this.

Usage:
    python speed_tracker.py --video traffic.mp4 --calib calibration.json --limit 80
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO ids)
SMOOTHING_WINDOW = 15           # frames used to average speed (reduces jitter)
MIN_FRAMES_FOR_SPEED = 5        # need at least this many points before trusting speed
LANE_STATS_WINDOW = 100         # how many recent speed samples to average per lane


class SpeedEstimator:
    def __init__(self, calib_path, fps, speed_limit_kmh=80):
        with open(calib_path, "r") as f:
            calib = json.load(f)

        src_pts = np.array(calib["homography_src"], dtype=np.float32)
        dst_pts = np.array(calib["homography_dst"], dtype=np.float32)
        self.H, _ = cv2.findHomography(src_pts, dst_pts)

        self.lanes = {
            name: np.array(poly, dtype=np.int32)
            for name, poly in calib.get("lanes", {}).items()
        }

        self.fps = fps
        self.speed_limit = speed_limit_kmh

        # track_id -> deque of (world_x, world_y, frame_idx)
        self.history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))
        # track_id -> last computed speed (for smoothing/hysteresis on the overlay)
        self.last_speed = {}
        # track_ids already logged as violators (avoid duplicate rows every frame)
        self.logged_violations = set()
        # lane name -> deque of recent speed readings, for the live per-lane stats overlay
        self.lane_speed_history = defaultdict(lambda: deque(maxlen=LANE_STATS_WINDOW))

    def pixel_to_world(self, px, py):
        p = np.array([px, py, 1.0])
        w = self.H @ p
        w /= w[2]
        return float(w[0]), float(w[1])

    def get_lane(self, cx, cy):
        for name, poly in self.lanes.items():
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                return name
        return "unknown"

    def update(self, track_id, bbox, frame_idx):
        """bbox = (x1, y1, x2, y2). Uses bottom-center point (road contact point)."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = y2  # bottom of box = point touching the road, best for homography

        wx, wy = self.pixel_to_world(cx, cy)
        self.history[track_id].append((wx, wy, frame_idx))

        speed_kmh = self.last_speed.get(track_id, 0.0)
        hist = self.history[track_id]
        if len(hist) >= MIN_FRAMES_FOR_SPEED:
            x_old, y_old, f_old = hist[0]
            x_new, y_new, f_new = hist[-1]
            dist_m = np.hypot(x_new - x_old, y_new - y_old)
            dt = (f_new - f_old) / self.fps
            if dt > 0:
                speed_kmh = (dist_m / dt) * 3.6
                self.last_speed[track_id] = speed_kmh

        lane = self.get_lane(cx, cy)

        # feed the per-lane rolling stats, but only once we trust the speed reading
        # and only for a real lane (skip "unknown" so it doesn't pollute lane stats)
        if len(hist) >= MIN_FRAMES_FOR_SPEED and lane != "unknown":
            self.lane_speed_history[lane].append(speed_kmh)

        is_violation = speed_kmh > self.speed_limit
        return speed_kmh, lane, is_violation, (int(cx), int(cy))

    def get_lane_stats(self):
        """Returns {lane_name: (avg_speed, sample_count)} for all lanes with data so far."""
        stats = {}
        for lane in self.lanes.keys():
            samples = self.lane_speed_history.get(lane)
            if samples:
                stats[lane] = (sum(samples) / len(samples), len(samples))
            else:
                stats[lane] = (0.0, 0)
        return stats


def draw_lanes(frame, lanes):
    overlay = frame.copy()
    for name, poly in lanes.items():
        cv2.polylines(overlay, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
        cx, cy = poly.mean(axis=0).astype(int)
        cv2.putText(overlay, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 200, 0), 2)
    return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)


def draw_stats_panel(frame, violation_count, lane_stats, speed_limit, corner="top-left", margin=15):
    """Draws a semi-transparent panel with a running violation counter and
    per-lane average speed stats, e.g.:

        Violations: 7
        --------------
        lane1: 42 km/h (n=88)
        lane2: 55 km/h (n=102)
    """
    h, w = frame.shape[:2]
    line_height = 26
    n_lines = 2 + len(lane_stats)  # counter line + separator + one per lane
    panel_w, panel_h = 260, line_height * n_lines + 20

    if corner == "top-left":
        x, y = margin, margin
    elif corner == "top-right":
        x, y = w - panel_w - margin, margin
    elif corner == "bottom-left":
        x, y = margin, h - panel_h - margin
    else:  # bottom-right
        x, y = w - panel_w - margin, h - panel_h - margin

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), (255, 255, 255), 1)

    ty = y + 24
    cv2.putText(frame, f"Violations: {violation_count}", (x + 12, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    ty += line_height

    cv2.line(frame, (x + 10, ty - 14), (x + panel_w - 10, ty - 14), (100, 100, 100), 1)

    for lane_name, (avg_speed, count) in lane_stats.items():
        text = f"{lane_name}: {avg_speed:.0f} km/h (n={count})"
        color = (0, 0, 255) if avg_speed > speed_limit else (255, 255, 255)
        cv2.putText(frame, text, (x + 12, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        ty += line_height

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file (or 0 for webcam)")
    parser.add_argument("--calib", default="calibration.json", help="Path to calibration JSON from calibrate.py")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--limit", type=float, default=80.0, help="Speed limit in km/h")
    parser.add_argument("--out", default="output.mp4", help="Path to save annotated output video")
    parser.add_argument("--log", default="violations.csv", help="CSV file to log violations")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview window")
    parser.add_argument("--stats-corner", default="top-left",
                         choices=["top-left", "top-right", "bottom-left", "bottom-right"],
                         help="Where to place the violation counter / lane stats panel")
    args = parser.parse_args()

    video_src = 0 if args.video == "0" else args.video
    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = YOLO(args.model)
    estimator = SpeedEstimator(args.calib, fps, speed_limit_kmh=args.limit)

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_exists = os.path.isfile(args.log)
    log_file = open(args.log, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["track_id", "lane", "speed_kmh", "frame_idx", "timestamp"])

    frame_idx = 0
    print("Processing... press 'q' to stop early (if display enabled).")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame, persist=True, classes=VEHICLE_CLASSES,
            tracker="bytetrack.yaml", verbose=False
        )

        annotated = draw_lanes(frame, estimator.lanes) if estimator.lanes else frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()

            for box, tid, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                speed_kmh, lane, violation, point = estimator.update(tid, (x1, y1, x2, y2), frame_idx)

                color = (0, 0, 255) if violation else (0, 255, 0)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID{tid} {lane} {speed_kmh:.0f}km/h"
                cv2.putText(annotated, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if violation and tid not in estimator.logged_violations:
                    estimator.logged_violations.add(tid)
                    log_writer.writerow([tid, lane, f"{speed_kmh:.1f}", frame_idx,
                                          time.strftime("%Y-%m-%d %H:%M:%S")])
                    log_file.flush()
                    os.fsync(log_file.fileno())
                    crop = frame[max(0,int(y1)):int(y2), max(0,int(x1)):int(x2)]
                    if crop.size > 0:
                        os.makedirs("violation_snapshots", exist_ok=True)
                        cv2.imwrite(f"violation_snapshots/track_{tid}_frame_{frame_idx}.jpg", crop)

        lane_stats = estimator.get_lane_stats()
        annotated = draw_stats_panel(
            annotated, len(estimator.logged_violations), lane_stats,
            estimator.speed_limit, corner=args.stats_corner
        )

        writer.write(annotated)
        if not args.no_display:
            cv2.imshow("Speed Estimation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    log_file.close()
    cv2.destroyAllWindows()
    print(f"Done. Annotated video: {args.out} | Violations log: {args.log}")


if __name__ == "__main__":
    main()