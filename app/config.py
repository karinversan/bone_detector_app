from __future__ import annotations

import os


WEIGHTS_DIR = "weights"
FRCNN_FILENAME = os.getenv("HF_FILENAME_FRCNN", "model_final_frcnn.pth")
YOLO_FILENAME = os.getenv("HF_FILENAME_YOLO", "best_yolo26m_640.pt")
DETECTRON_DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/{FRCNN_FILENAME}"
YOLO_DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/{YOLO_FILENAME}"
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

FRCNN_VAL_JSON = "data/BoneFractureYolo8/val_merged.json"
FRCNN_VAL_IMAGES = "data/BoneFractureYolo8/valid/images"
FRCNN_VAL_LABELS = "data/BoneFractureYolo8/valid/labels"
YOLO_DATA_YAML = "data/BoneFractureYolo8/data.yaml"
METRICS_CACHE_DIR = "cache"
FRCNN_METRICS_CACHE = f"{METRICS_CACHE_DIR}/metrics_frcnn.json"
YOLO_METRICS_CACHE = f"{METRICS_CACHE_DIR}/metrics_yolo.json"
METRICS_LOCK = f"{METRICS_CACHE_DIR}/metrics.lock"
FROC_IOU = 0.5
FROC_FP_PER_IMAGE = [0.5, 1.0, 2.0, 4.0]

NEW_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]

OLD_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]

MERGE_FROM_OLD = 3
MERGE_TO_OLD = 4
KEPT_OLD_IDS = [0, 1, 2, 4, 5, 6]
OLD_TO_NEW = {old_id: new_id for new_id, old_id in enumerate(KEPT_OLD_IDS)}

FRCNN_TRAINING_ATTRS = {
    "model": "Faster R-CNN ResNet-50-FPN",
    "image_size": 1024,
    "anchor_ratios": "[0.33, 0.5, 1.0, 2.0, 3.0] * 5",
    "ims_per_batch": 2,
    "base_lr": 0.0025,
    "warmup_iters": 500,
    "max_iter": 15000,
    "steps": "(10000, 13000)",
    "gamma": 0.1,
    "detections_per_image": 300,
}

YOLO_TRAINING_ATTRS = {
    "model": "YOLO26m",
    "image_size": 640,
    "epochs": 50,
}
