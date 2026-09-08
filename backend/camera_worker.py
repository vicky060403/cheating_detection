"""
camera_worker.py

One background thread per camera: captures frames, runs YOLO detection,
draws boxes, keeps the latest frame ready as JPEG bytes for MJPEG streaming,
and fires an on_alert(...) callback when "cheating" is confirmed over
several consecutive frames.
"""

import os
import csv
import time
import threading
from datetime import datetime

import cv2


def test_camera_source(source):
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            return False
        ok, _ = capture.read()
        return ok
    finally:
        capture.release()


class CameraWorker(threading.Thread):
    def __init__(self, name, source, model, model_lock, config, on_alert=None):
        super().__init__(daemon=True)
        self.name = name
        self.source = source
        self.model = model
        self.model_lock = model_lock
        self.cfg = config
        self.on_alert = on_alert  # callable(alert_dict)

        self.cap = cv2.VideoCapture(source)
        self.frame_lock = threading.Lock()
        self.latest_jpeg = None

        self.running = True
        self.connected = False
        self.alert_active = False
        self.consecutive_cheat_frames = 0
        self.last_alert_time = 0.0

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.connected = False
                time.sleep(1.0)
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                continue
            self.connected = True

            h, w = frame.shape[:2]
            fw = self.cfg["frame_width"]
            frame = cv2.resize(frame, (fw, int(h * fw / w)))

            with self.model_lock:
                result = self.model.predict(
                    frame,
                    conf=self.cfg["conf_threshold"],
                    iou=self.cfg["iou_threshold"],
                    verbose=False,
                )[0]

            cheating_this_frame = False
            best_conf = 0.0

            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                is_cheat = cls_name.lower() == self.cfg["cheating_class_name"].lower()
                color = (0, 0, 255) if is_cheat else (0, 200, 0)
                if is_cheat:
                    cheating_this_frame = True
                    best_conf = max(best_conf, conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{cls_name} {conf:.2f}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

            self.consecutive_cheat_frames = (
                self.consecutive_cheat_frames + 1 if cheating_this_frame else 0
            )
            confirmed = self.consecutive_cheat_frames >= self.cfg["consecutive_frames_to_confirm"]
            self.alert_active = confirmed

            if confirmed:
                now = time.time()
                if now - self.last_alert_time > self.cfg["alert_cooldown_seconds"]:
                    self._trigger_alert(frame, best_conf)
                    self.last_alert_time = now
                cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 6)
                cv2.putText(
                    frame, "CHEATING ALERT", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                )

            cv2.putText(frame, self.name, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            ok2, buf = cv2.imencode(".jpg", frame)
            if ok2:
                with self.frame_lock:
                    self.latest_jpeg = buf.tobytes()

    def _trigger_alert(self, frame, confidence):
        ts_dt = datetime.now()
        ts = ts_dt.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{ts}.jpg"
        path = os.path.join(self.cfg["snapshot_dir"], filename)
        cv2.imwrite(path, frame)

        alert = {
            "camera": self.name,
            "confidence": round(confidence, 2),
            "timestamp": ts_dt.isoformat(timespec="seconds"),
            "snapshot_file": filename,
        }
        self._log_csv(alert, path)
        if self.on_alert:
            self.on_alert(alert)

    def _log_csv(self, alert, path):
        log_file = self.cfg["log_file"]
        write_header = not os.path.exists(log_file)
        with open(log_file, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "camera", "confidence", "snapshot_path"])
            w.writerow([alert["timestamp"], alert["camera"], alert["confidence"], path])

    def get_jpeg(self):
        with self.frame_lock:
            return self.latest_jpeg

    def stop(self):
        self.running = False
        self.cap.release()
