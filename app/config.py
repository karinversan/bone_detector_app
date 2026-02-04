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
    "config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "datasets": "fracture_train_merged / fracture_val_merged",
    "labels_source": "YOLO seg -> COCO (merged classes)",
    "num_classes": 6,
    "input_size": "1024 (min/max)",
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
    "config": "Ultralytics YOLO (from checkpoint)",
    "datasets": "not provided",
    "input_size": "not provided",
    "num_classes": "from checkpoint",
}
