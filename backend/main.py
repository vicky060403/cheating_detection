"""
main.py

FastAPI backend for the real-time exam cheating detection system.

Endpoints:
  GET  /api/cameras         -> list cameras + live status
  POST /api/cameras         -> add a camera at runtime {"name": ..., "source": ...}
  DELETE /api/cameras/{name}-> stop and remove a camera
  GET  /video_feed/{name}   -> MJPEG live stream for one camera
  GET  /api/alerts          -> recent alert history (JSON)
  WS   /ws/alerts           -> live push of new alerts
  GET  /snapshots/{file}    -> saved alert snapshot images
  GET  /                    -> the frontend dashboard (static files)

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import asyncio
import threading
from typing import Dict, List

import torch
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from camera_worker import CameraWorker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

with open(os.path.join(BASE_DIR, "config.json")) as f:
    CONFIG = json.load(f)

CONFIG["snapshot_dir"] = os.path.join(BASE_DIR, "cheating_alerts", "snapshots")
CONFIG["log_file"] = os.path.join(BASE_DIR, "cheating_alerts", "cheating_log.csv")
os.makedirs(CONFIG["snapshot_dir"], exist_ok=True)

app = FastAPI(title="Exam Cheating Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.mount("/snapshots", StaticFiles(directory=CONFIG["snapshot_dir"]), name="snapshots")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model '{CONFIG['model_path']}' on device: {device}")
model = YOLO(CONFIG["model_path"])
model.to(device)
model_lock = threading.Lock()

workers: Dict[str, CameraWorker] = {}
alerts_history: List[dict] = []
MAX_HISTORY = 200

connected_ws: List[WebSocket] = []
main_loop = None  # asyncio event loop, set on startup


def broadcast_alert(alert: dict):
    """Called from a CameraWorker thread when cheating is confirmed."""
    alerts_history.insert(0, alert)
    del alerts_history[MAX_HISTORY:]
    if main_loop is not None:
        asyncio.run_coroutine_threadsafe(_broadcast(alert), main_loop)


async def _broadcast(alert: dict):
    dead = []
    for ws in connected_ws:
        try:
            await ws.send_json({"type": "alert", "data": alert})
        except Exception:
            dead.append(ws)
    for d in dead:
        if d in connected_ws:
            connected_ws.remove(d)


def start_camera(name: str, source):
    worker = CameraWorker(name, source, model, model_lock, CONFIG, on_alert=broadcast_alert)
    worker.start()
    workers[name] = worker


@app.on_event("startup")
async def startup():
    global main_loop
    main_loop = asyncio.get_event_loop()
    for name, source in CONFIG.get("cameras", {}).items():
        start_camera(name, source)


@app.on_event("shutdown")
async def shutdown():
    for worker in workers.values():
        worker.stop()


@app.get("/api/cameras")
def list_cameras():
    return [
        {"name": name, "connected": w.connected, "alert_active": w.alert_active}
        for name, w in workers.items()
    ]


@app.post("/api/cameras")
def add_camera(payload: dict = Body(...)):
    name = payload.get("name")
    source = payload.get("source")
    if not name or source is None:
        return JSONResponse({"error": "name and source are required"}, status_code=400)
    if isinstance(source, str) and source.strip().isdigit():
        source = int(source.strip())
    if name in workers:
        return JSONResponse({"error": "camera already exists"}, status_code=400)
    start_camera(name, source)
    return {"status": "started", "name": name}


@app.delete("/api/cameras/{name}")
def remove_camera(name: str):
    worker = workers.pop(name, None)
    if not worker:
        return JSONResponse({"error": "camera not found"}, status_code=404)
    worker.stop()
    return {"status": "stopped", "name": name}


def mjpeg_generator(worker: CameraWorker):
    while worker.running:
        frame = worker.get_jpeg()
        if frame is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


@app.get("/video_feed/{name}")
def video_feed(name: str):
    worker = workers.get(name)
    if not worker:
        return JSONResponse({"error": "camera not found"}, status_code=404)
    return StreamingResponse(
        mjpeg_generator(worker), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    return alerts_history[:limit]


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    await websocket.accept()
    connected_ws.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive ping from client; content ignored
    except WebSocketDisconnect:
        if websocket in connected_ws:
            connected_ws.remove(websocket)


# Serve the frontend dashboard last, so it doesn't shadow the API routes above.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
