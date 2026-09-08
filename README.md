# Real-Time Exam Cheating Detection

A YOLO-based system that watches multiple exam-room cameras live, flags
cheating behavior, and shows it on a web dashboard.

```
project/
├── train_cheating_model.py   # trains the YOLO model on the Roboflow dataset
├── requirements-training.txt
├── backend/                  # FastAPI server: camera capture, inference, alerts API
│   ├── main.py
│   ├── camera_worker.py
│   ├── config.json
│   └── requirements.txt
└── frontend/                 # Static web dashboard (served by the backend)
    ├── index.html
    ├── app.js
    └── styles.css
```

## 1. Train the model

```bash
pip install -r requirements-training.txt
python train_cheating_model.py
```

This asks for your Roboflow API key, downloads the "Graduation Project"
dataset (classes: `Cheating`, `Normal`), and trains a YOLO11 model. It
prints the path to the resulting `best.pt` and the validation
precision/recall/F1.

Copy that `best.pt` into `backend/` (or point `model_path` in
`backend/config.json` at it).

## 2. Configure cameras

Edit `backend/config.json`:

```json
{
  "model_path": "best.pt",
  "cheating_class_name": "Cheating",
  "conf_threshold": 0.45,
  "iou_threshold": 0.5,
  "consecutive_frames_to_confirm": 5,
  "alert_cooldown_seconds": 10,
  "frame_width": 640,
  "cameras": {
    "Camera-1": 0,
    "Camera-2": "rtsp://user:pass@192.168.1.11:554/stream1",
    "Camera-3": "videos/hall_row3.mp4"
  }
}
```

Sources can be a webcam index, an RTSP/IP camera URL, or a video file path
(useful for testing without physical cameras). You can also add/remove
cameras later from the dashboard itself, without restarting the server.

For multiple classroom CCTV cameras, use one fixed name and RTSP URL per
camera. Connect the cameras to the same network as the backend computer,
enable RTSP on each camera, and test each URL in VLC first. The dashboard's
**Add Camera** dialog can test an RTSP URL before adding it. Cameras added
from the dashboard are saved to `backend/config.json` and restored after a
backend restart. A PoE switch is recommended for wired IP cameras; an NVR is
optional.

## 3. Run the backend (serves the API + the dashboard)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser — that's the dashboard,
served directly by the backend (no separate frontend server needed).

## What the dashboard does

- **Live Cameras panel**: one tile per camera, streamed live (MJPEG). A
  camera's tile gets a red outline the moment cheating is confirmed.
- **Add Camera** button: register a new camera source at runtime.
- **Cheating Alerts panel**: live feed of confirmed alerts, pushed instantly
  over a WebSocket, each with its snapshot thumbnail, camera name,
  confidence, and time. Reloading the page re-fetches recent history from
  `/api/alerts`.
- Every alert is also written to `backend/cheating_alerts/cheating_log.csv`
  and its snapshot saved to `backend/cheating_alerts/snapshots/`.

An alert only fires after `consecutive_frames_to_confirm` frames in a row
detect "Cheating" (filters out one-off false positives), and a camera won't
re-alert for `alert_cooldown_seconds` afterward.

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cameras` | List cameras with connection/alert status |
| POST | `/api/cameras` | Add a camera: `{"name": "...", "source": "..."}` |
| DELETE | `/api/cameras/{name}` | Stop and remove a camera |
| GET | `/video_feed/{name}` | MJPEG live stream for one camera |
| GET | `/api/alerts?limit=50` | Recent alert history (JSON) |
| WS | `/ws/alerts` | Live push of new alerts |
| GET | `/snapshots/{file}` | Saved alert snapshot image |

## Tuning tips

| Problem | What to change |
|---|---|
| Too many false alerts | Raise `conf_threshold` and/or `consecutive_frames_to_confirm` in `config.json` |
| Missing real cheating events | Lower `conf_threshold`, retrain with more/varied data, or lower `consecutive_frames_to_confirm` |
| Low FPS with many cameras | Run on a GPU (auto-detected), lower `frame_width`, or use a smaller base model when training (`yolo11n.pt`/`yolo11s.pt`) |
| RTSP camera keeps dropping | The worker auto-reconnects after 1s; increase that delay in `camera_worker.py` if your camera needs longer |

## Notes

- All camera threads share one loaded YOLO model behind a lock, which is
  memory-efficient but serializes inference across cameras. If you have
  GPU memory to spare and want higher combined FPS, give each
  `CameraWorker` its own model instance instead.
- This is a per-frame visual detector, not proof of misconduct — treat
  alerts and snapshots as leads for a human proctor to review, not as
  automatic verdicts.
