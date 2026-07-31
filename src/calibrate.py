"""
Calibration helper for speed_tracker.py

Step 1: Click 4 points in the image that correspond to a known real-world
        rectangle on the road (e.g. corners of a marked box, or two lane-marking
        segments a known distance apart). Press 'h' to lock them in as the
        homography source points, then enter the real-world width/height (meters).

Step 2: Draw one or more lane polygons (click points, press 'n' to finish a
        lane and start a new one, press 'l' when done with all lanes).

Step 3: Press 's' to save everything to calibration.json.

Usage:
    python calibrate.py --video traffic.mp4
    python calibrate.py --image sample_frame.jpg
"""

import argparse
import json

import cv2
import numpy as np

MODE_HOMOGRAPHY = "homography"
MODE_LANES = "lanes"


class Calibrator:
    def __init__(self, frame):
        self.frame = frame
        self.display = frame.copy()
        self.mode = MODE_HOMOGRAPHY

        self.homography_src = []      # 4 clicked points (pixels)
        self.real_width = None
        self.real_height = None

        self.lanes = {}               # name -> list of points
        self.current_lane_pts = []
        self.lane_count = 0

    def redraw(self):
        self.display = self.frame.copy()

        # draw homography points
        for i, pt in enumerate(self.homography_src):
            cv2.circle(self.display, pt, 6, (0, 0, 255), -1)
            cv2.putText(self.display, str(i + 1), (pt[0] + 8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(self.homography_src) >= 2:
            for i in range(len(self.homography_src)):
                p1 = self.homography_src[i]
                p2 = self.homography_src[(i + 1) % len(self.homography_src)]
                if i < len(self.homography_src) - 1 or len(self.homography_src) == 4:
                    cv2.line(self.display, p1, p2, (0, 0, 255), 1)

        # draw saved lanes
        colors = [(255, 200, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0)]
        for idx, (name, pts) in enumerate(self.lanes.items()):
            color = colors[idx % len(colors)]
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(self.display, [arr], isClosed=True, color=color, thickness=2)
            cx, cy = arr.mean(axis=0).astype(int)
            cv2.putText(self.display, name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # draw current in-progress lane
        if self.current_lane_pts:
            arr = np.array(self.current_lane_pts, dtype=np.int32)
            cv2.polylines(self.display, [arr], isClosed=False, color=(0, 165, 255), thickness=2)
            for pt in self.current_lane_pts:
                cv2.circle(self.display, pt, 4, (0, 165, 255), -1)

        mode_text = f"MODE: {self.mode.upper()}  |  h=homography mode, l=lane mode, n=next lane, s=save, q=quit"
        cv2.putText(self.display, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2)

    def on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == MODE_HOMOGRAPHY:
            if len(self.homography_src) < 4:
                self.homography_src.append((x, y))
            else:
                print("Already have 4 homography points. Press 'h' again to reset if needed.")
        elif self.mode == MODE_LANES:
            self.current_lane_pts.append((x, y))

        self.redraw()

    def finish_lane(self):
        if len(self.current_lane_pts) >= 3:
            self.lane_count += 1
            name = f"lane{self.lane_count}"
            self.lanes[name] = self.current_lane_pts
            print(f"Saved {name} with {len(self.current_lane_pts)} points.")
        else:
            print("Need at least 3 points to form a lane polygon.")
        self.current_lane_pts = []

    def save(self, path, real_width, real_height):
        # destination rectangle in meters, scaled up for numeric stability
        dst = [[0, 0], [real_width, 0], [real_width, real_height], [0, real_height]]
        data = {
            "homography_src": self.homography_src,
            "homography_dst": dst,
            "lanes": self.lanes,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video; first frame is used")
    parser.add_argument("--image", help="Path to a still image instead of video")
    parser.add_argument("--out", default="calibration.json")
    args = parser.parse_args()

    if args.image:
        frame = cv2.imread(args.image)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Could not read first frame from video.")
    else:
        raise ValueError("Provide --video or --image")

    calib = Calibrator(frame)
    calib.redraw()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", calib.on_click)

    print("\nSTEP 1: Click 4 points forming a known rectangle on the road (in order: "
          "top-left, top-right, bottom-right, bottom-left of the real-world rectangle).")
    print("STEP 2: Press 'l' to switch to lane-drawing mode, click polygon points per lane, "
          "press 'n' after each lane. Press 'l' again to go back to homography mode if needed.")
    print("STEP 3: Press 's' to save.\n")

    while True:
        calib.redraw()
        cv2.imshow("Calibration", calib.display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("h"):
            calib.mode = MODE_HOMOGRAPHY
            calib.homography_src = []
            print("Reset homography points. Click 4 new points.")
        elif key == ord("l"):
            calib.mode = MODE_LANES
            print("Lane mode. Click points for the current lane polygon.")
        elif key == ord("n"):
            calib.finish_lane()
        elif key == ord("s"):
            if len(calib.homography_src) != 4:
                print("Need exactly 4 homography points before saving.")
                continue
            try:
                real_width = float(input("Enter real-world WIDTH of the rectangle in meters: "))
                real_height = float(input("Enter real-world HEIGHT (length) of the rectangle in meters: "))
            except ValueError:
                print("Invalid number, try again.")
                continue
            calib.save(args.out, real_width, real_height)
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
