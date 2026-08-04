# Multi-Lane Vehicle Speed Estimation — Starter Kit

## Files
- `speed_tracker.py` — main pipeline: detection (YOLO) + tracking (ByteTrack) + speed estimation + violation logging
- `calibrate.py` — interactive tool to set up homography points and lane polygons for a specific camera angle
- `requirements.txt` — dependencies 

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The first run of `speed_tracker.py` will auto-download `yolov8n.pt`.

## Step 1: Calibrate the camera

You need a real-world reference. The easiest option: measure a rectangle on
the road itself (e.g. distance between two lane-marking dashes, or a painted
box) with a tape measure or known road standard (NHA lane width ≈ 3.5 m).

```bash
python calibrate.py --video sample_traffic.mp4 --out calibration.json
```

- Click the 4 corners of your reference rectangle **in order**: top-left,
  top-right, bottom-right, bottom-left (as seen in the camera image — "top"
  meaning farther from the camera).
- Press `l` to switch to lane mode, click each lane's polygon corners, press
  `n` after each lane, repeat for every lane.
- Press `s` and enter the real width/height of your reference rectangle in
  meters when prompted.
- Press `q` to quit once saved.

This produces `calibration.json` — inspect/edit it by hand any time, it's
plain JSON:

```json
{
  "homography_src": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "homography_dst": [[0,0],[3.5,0],[3.5,10],[0,10]],
  "lanes": {
    "lane1": [[x,y], [x,y], [x,y]],
    "lane2": [[x,y], [x,y], [x,y]]
  }
}
```

## Step 2: Run speed estimation

```bash
python speed_tracker.py --video sample_traffic.mp4 --calib calibration.json --limit 80
```

Outputs:
- `output.mp4` — annotated video (green boxes = normal, red = over the limit)
- `violations.csv` — one row per violating vehicle (`track_id, lane, speed_kmh, frame_idx, timestamp`)
- `violation_snapshots/` — cropped image of each violating vehicle

Useful flags:
- `--model yolov8s.pt` — bigger/more accurate model (slower)
- `--no-display` — headless mode, e.g. running on a server/Pi
- `--video 0` — use a live webcam instead of a file

## Accuracy tips

1. **Camera angle**: an elevated, angled-down view (like a pole-mounted CCTV)
   gives far better homography accuracy than a straight dashcam angle.
2. **Motorcycles**: they weave between lanes constantly — don't be surprised
   if their lane label flickers between lane1/lane2; this is expected, not a bug.
3. **Reference rectangle size**: bigger reference rectangles (spanning more of
   the road) generally give more stable homography than small ones.
4. **Custom classes**: if you want rickshaws/Suzukis as a separate detected
   class, label ~500-1000 images (Roboflow is fast for this) and fine-tune
   YOLO with `model.train(data="your_data.yaml", epochs=50)`.
5. **Smoothing window**: `SMOOTHING_WINDOW` and `MIN_FRAMES_FOR_SPEED` at the
   top of `speed_tracker.py` control how many frames are averaged before
   trusting a speed reading — raise these if speeds look jittery, lower them
   if response feels too laggy on fast-moving traffic.

## Next steps (stretch goals)
- Add ANPR: crop plates from `violation_snapshots/`, run PaddleOCR/EasyOCR,
  fine-tune on local plate samples (Pakistani plates vary in font/format).
- Add a database (SQLite/Postgres) instead of CSV for the violation log.
