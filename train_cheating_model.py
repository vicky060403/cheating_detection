"""
    Train a YOLO model on the exam-cheating Roboflow dataset.
    Usage:
        python train_cheating_model.py
    Notes:
    - Make sure you have a Roboflow API key set in your environment:
        export ROBOFLOW_API_KEY="your_api_key_here"
    - This script will download the dataset, train a YOLO model, and validate it.
    - The best weights will be saved in the PROJECT_DIR/RUN_NAME directory.

"""
import os
import getpass
import torch
from roboflow import Roboflow
from ultralytics import YOLO
from pathlib import Path

# configurable parameters for training

ROBOFLOW_WORKSPACE = "omars-workspace-opfmx"
ROBOFLOW_PROJECT = "graduation-project-jewvv"
ROBOFLOW_VERSION = 5
DATASET_FORMAT = "yolov11"

BASE_MODEL = "yolo11m.pt"
RESUME_FROM = None  
IMG_SIZE = 640
EPOCHS = 100
# PATIENCE = 30        # stop early if val mAP doesn't improve for this many epochs
BATCH_SIZE = -1        # -1 = auto-pick the largest batch that fits in GPU memory

## Augmentation settings for YOLOv11 training. See https://docs.roboflow.com/roboflow-train/augmentation for details.
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

PROJECT_DIR = os.environ.get("TRAIN_OUTPUT_DIR", "runs_cheating_detection")
RUN_NAME = "yolo11l_v5"


## check if GPU is available and print its name
def check_gpu():
    available = torch.cuda.is_available()
    print(f"GPU Available: {available}")
    if available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU detected. On Lightning AI, make sure your Studio has a GPU attached.")
    return available

## download the dataset from Roboflow and return the path to the data.yaml file
def download_dataset():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(DATASET_FORMAT)
    print(f"Dataset downloaded to: {dataset.location}")
    return f"{dataset.location}/data.yaml"


## Train a YOLO model on the dataset and return the trained model and results
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


## Validate the trained model on the validation set
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

## Main entry point for the script
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
