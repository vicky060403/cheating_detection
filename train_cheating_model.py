"""
train_cheating_model.py

Trains a YOLO model on the exam-cheating Roboflow dataset. Tuned to be run
on a Lightning AI Studio GPU (A10G / L4 / A100) for a longer, higher-accuracy
run than a free-tier Colab session would comfortably allow.

Usage (from a Lightning Studio terminal):
    pip install -r requirements-training.txt
    python train_cheating_model.py

For a run that survives you closing the browser tab, launch it with nohup:
    nohup python train_cheating_model.py > train.log 2>&1 &
    tail -f train.log
"""

import os
import getpass
import torch
from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path

# ================= CONFIG =================

ROBOFLOW_WORKSPACE = "omars-workspace-opfmx"
ROBOFLOW_PROJECT = "graduation-project-jewvv"
ROBOFLOW_VERSION = 5
DATASET_FORMAT = "yolov11"

# Bigger backbone than the notebook's yolo11m -- worth it now that you have a
# real GPU budget on Lightning AI. Drop to "yolo11m.pt" if training is too
# slow for your chosen GPU tier, or go to "yolo11x.pt" on an A100.
BASE_MODEL = "yolo11m.pt"
RESUME_FROM = None  # e.g. "runs/detect/train/weights/last.pt" to resume a previous run

IMG_SIZE = 640
EPOCHS = 100
# PATIENCE = 30        # stop early if val mAP doesn't improve for this many epochs
BATCH_SIZE = -1        # -1 = auto-pick the largest batch that fits in GPU memory

# Augmentation -- the current model misses ~40% of "Cheating" instances and has
# a noticeable false-positive rate on background, so more aggressive
# augmentation (rotation/scale/color jitter + mosaic/mixup) should help it
# generalize past the exact camera angles/lighting in the training set.
AUGMENTATION = dict(
    mosaic=1.0,
    mixup=0.1,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    close_mosaic=15,   # disable mosaic for the last N epochs to sharpen boxes
)

# On Lightning AI, save into persistent storage so runs survive studio restarts.
PROJECT_DIR = os.environ.get("TRAIN_OUTPUT_DIR", "runs_cheating_detection")
RUN_NAME = "yolo11l_v5"

# ============================================


def check_gpu():
    available = torch.cuda.is_available()
    print(f"GPU Available: {available}")
    if available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU detected. On Lightning AI, make sure your Studio has a GPU attached.")
    return available


def download_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DATASET_FORMAT)
    print(f"Dataset downloaded to: {dataset.location}")
    return f"{dataset.location}/data.yaml"


def train(data_yaml_path):
    if RESUME_FROM:
        model = YOLO(RESUME_FROM)
        results = model.train(resume=True)
    else:
        model = YOLO(BASE_MODEL)
        results = model.train(
            data=data_yaml_path,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_DIR,
            name=RUN_NAME,
            cos_lr=True,
            **AUGMENTATION,
        )
    return model, results


def validate(best_weights_path, data_yaml_path):
    model = YOLO(best_weights_path)
    metrics = model.val(data=data_yaml_path, imgsz=IMG_SIZE, batch=16, conf=0.25, iou=0.7, plots=True)
    mp, mr = metrics.box.mp, metrics.box.mr
    f1 = 2 * (mp * mr) / (mp + mr + 1e-9)
    print("\nValidation results:")
    print(f"  Precision:     {mp:.4f}")
    print(f"  Recall:        {mr:.4f}")
    print(f"  F1-score:      {f1:.4f}")
    print(f"  mAP50:         {metrics.box.map50:.4f}")
    print(f"  mAP50-95:      {metrics.box.map:.4f}")
    return metrics


if __name__ == "__main__":
    check_gpu()
    data_yaml = download_dataset()
    model, results = train(data_yaml)

    best_path = str(model.trainer.best)
    print(f"\nTraining complete. Best weights saved at: {best_path}")

    validate(best_path, data_yaml)

    print(
        "\nNext step: copy this best.pt file into backend/ "
        "(or update model_path in backend/config.json)."
    )
