from __future__ import annotations

import numpy as np
import torch


def load_yolo_model(weights_path: str):
    try:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None
    except ImportError as exc:
        raise RuntimeError(
            f"Ultralytics import failed: {exc}"
        ) from exc
    if safe_globals is not None:
        with safe_globals([DetectionModel]):
            return YOLO(weights_path)
    try:
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
    return YOLO(weights_path)


def resolve_yolo_label(model, class_names: list[str], cls_id: int) -> str:
    if class_names and 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    if hasattr(model, "names"):
        names = model.names
        if isinstance(names, dict) and cls_id in names:
            return str(names[cls_id])
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
    return f"class_{cls_id}"


def predict_yolo(model, image, class_names: list[str]) -> list[dict]:
    results = model.predict(source=np.array(image), verbose=False)
    result = results[0]
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        label = resolve_yolo_label(model, class_names, cls_id)
        detections.append(
            {
                "label": label,
                "class_id": int(cls_id),
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return detections
