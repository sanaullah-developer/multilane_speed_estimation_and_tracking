# Multi-Lane Vehicle Speed Estimation & Violation Detection

An end-to-end computer vision pipeline that detects and tracks vehicles across multiple lanes from a single fixed camera, estimates real-world speed using homography-based perspective correction, flags speed violations automatically, and surfaces everything through a live monitoring dashboard.

No radar gun, no manual measurement — just a camera, calibration, and math.

## Demo

See `demo/` for a sample output video showing detection, tracking, per-lane speed overlay, and violation flagging in action.

## How it works

```
Video feed → YOLOv8 detection → ByteTrack tracking → homography-based
pixel-to-meters conversion → smoothed speed estimate → lane assignment
→ violation logging → live dashboard
```

The core problem this solves: a camera looking down a road distorts perspective — vehicles far from the camera move fewer pixels per frame than the same vehicle up close, even at identical real speed. Naively measuring pixel displacement gives systematically wrong speeds. This project uses **homography** (a perspective-correction transform) to convert every tracked vehicle's pixel position into real-world meters, frame by frame, so speed is calculated from actual distance traveled — not raw pixel movement.

## Repository structure

```
├── src/
│   ├── speed_tracker.py    # main pipeline: detection + tracking + speed + violations
│   ├── calibrate.py        # interactive homography + lane calibration tool
│   └── dashboard.py        # live Streamlit monitoring dashboard
├── config/
│   └── calibration.json    # example calibration output
├── demo/
│   └── ...                 # sample output video
├── experiments/
│   └── training_results.md # exploratory fine-tuning writeup (not used in production — see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The first run of `speed_tracker.py` will auto-download `yolov8n.pt` (stock YOLOv8, COCO-pretrained — detects car, motorcycle, bus, truck).

## Step 1 — Calibrate the camera

You need a real-world reference: measure a rectangle on the road itself (e.g. distance between two lane-marking dashes, or a painted box) with a tape measure or a known road standard.

```bash
python src/calibrate.py --video sample_traffic.mp4 --out config/calibration.json
```

- Click the 4 corners of your reference rectangle **in order**: top-left, top-right, bottom-right, bottom-left (as seen in the camera image — "top" meaning farther from the camera).
- Press `l` to switch to lane mode, click each lane's polygon corners, press `n` after each lane, and repeat for every lane on the road.
- Press `s` and enter the real width/height of your reference rectangle in meters when prompted.
- Press `q` to quit once saved.

This produces `calibration.json`:

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

## Step 2 — Run speed estimation

```bash
python src/speed_tracker.py --video sample_traffic.mp4 --calib config/calibration.json --limit 50
```

Outputs:
- `output.mp4` — annotated video (green boxes = normal, red = over the limit), with a live on-screen violation counter and per-lane average speed panel
- `violations.csv` — one row per violating vehicle (`track_id, lane, speed_kmh, frame_idx, timestamp`), flushed to disk immediately so it can be read live
- `violation_snapshots/` — cropped image of each violating vehicle

Useful flags:
- `--model yolov8s.pt` — bigger/more accurate model (slower)
- `--no-display` — headless mode, e.g. running on a server, or alongside the dashboard
- `--video 0` — use a live webcam instead of a file
- `--stats-corner top-left|top-right|bottom-left|bottom-right` — position of the violation counter / lane stats panel

## Step 3 — Run the live dashboard

While `speed_tracker.py` is running (or after it finishes), launch the dashboard in a separate terminal to monitor violations in real time:

```bash
streamlit run src/dashboard.py -- --log violations.csv --snapshots violation_snapshots --refresh 5
```

The dashboard shows:
- KPI summary — total violations, average violation speed, highest speed recorded, busiest lane
- Violations-per-lane bar chart
- Speed distribution histogram
- Violations-over-time timeline
- A live gallery of recent violation snapshots
- Full sortable violation log table

It auto-refreshes every `--refresh` seconds by reading directly from `violations.csv` and `violation_snapshots/` — run `speed_tracker.py --no-display` in one terminal and the dashboard in another (open in your browser) to watch violations populate live as they're detected.

## Accuracy tips

- **Camera angle:** an elevated, angled-down view (like a pole-mounted CCTV) gives far better homography accuracy than a straight dashcam angle.
- **Motorcycles:** they weave between lanes constantly — don't be surprised if their lane label flickers between lanes; this is expected, not a bug.
- **Reference rectangle size:** bigger reference rectangles (spanning more of the road) generally give more stable homography than small ones.
- **Smoothing window:** `SMOOTHING_WINDOW` and `MIN_FRAMES_FOR_SPEED` at the top of `speed_tracker.py` control how many frames are averaged before trusting a speed reading — raise these if speeds look jittery, lower them if response feels too laggy on fast-moving traffic.
- **Validate before trusting the numbers:** before relying on any reported speed, test with a vehicle at a known steady speed (speedometer- or GPS-confirmed) and compare against what the pipeline reports.

## Known limitations

- Runs on the stock YOLOv8 model (car/motorcycle/bus/truck). Local vehicle types outside the COCO class set — rickshaws, qingqi-rickshaws, loaders — are not detected. A custom fine-tuning pass was explored on a local-vehicle dataset; see `experiments/training_results.md` for the full writeup, including a diagnosed motorcycle-detection gap and confusion-matrix analysis. It's not currently used in the pipeline.
- Detection reliability drops for small, distant, or occluded vehicles — a known hard problem in dense traffic that would likely need higher inference resolution to meaningfully improve.
- Homography calibration is camera-angle-specific — recalibration is required for each new camera position.
- License plate recognition was scoped but not implemented: at the distances needed for full-road lane coverage, plates are frequently too low-resolution for reliable OCR with a single wide-FOV camera. Production ANPR systems typically solve this with a second, dedicated zoomed camera.

## Tech stack

YOLOv8 (Ultralytics), ByteTrack, OpenCV, Streamlit, Plotly, Pandas

## Next steps

- License plate cropping + OCR (PaddleOCR/EasyOCR), contingent on higher-resolution capture
- Database backend (SQLite/Postgres) instead of CSV for the violation log
- Revisit the custom fine-tuned model if motorcycle recall can be meaningfully improved (see `experiments/training_results.md`)