from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import NEW_NAMES


def _infer_detectron_ckpt_shapes(weights_path: Path) -> tuple[int | None, int | None]:
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    num_classes = None
    num_anchors = None

    cls_weight = state_dict.get("roi_heads.box_predictor.cls_score.weight")
    if cls_weight is not None and hasattr(cls_weight, "shape"):
        num_classes = int(cls_weight.shape[0]) - 1

    anchor_weight = state_dict.get("proposal_generator.rpn_head.anchor_deltas.weight")
    if anchor_weight is not None and hasattr(anchor_weight, "shape"):
        num_anchors = int(anchor_weight.shape[0] // 4)

    return num_classes, num_anchors


def _anchor_ratios_for_count(count: int | None) -> list[float]:
    if count == 5:
        return [0.33, 0.5, 1.0, 2.0, 3.0]
    if count == 3:
        return [0.5, 1.0, 2.0]
    if count is None or count <= 0:
        return [0.5, 1.0, 2.0]
    return [1.0] * count


def build_detectron_cfg(weights_path: str):
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed. Install detectron2 to use the Faster R-CNN model."
        ) from exc

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    num_classes_from_ckpt, num_anchors = _infer_detectron_ckpt_shapes(
        Path(weights_path)
    )
    num_classes = num_classes_from_ckpt or len(NEW_NAMES)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    anchor_ratios = _anchor_ratios_for_count(num_anchors)
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [anchor_ratios] * 5

    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = weights_path
    return cfg


def load_detectron_model(weights_path: str):
    try:
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed. Install detectron2 to use the Faster R-CNN model."
        ) from exc
    cfg = build_detectron_cfg(weights_path)
    return DefaultPredictor(cfg)


def predict_detectron(predictor, image, class_names: list[str]) -> list[dict]:
    image_bgr = np.array(image)[:, :, ::-1]
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy().astype(int)

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        label = (
            class_names[cls_id]
            if 0 <= cls_id < len(class_names)
            else f"class_{cls_id}"
        )
        detections.append(
            {
                "label": label,
                "class_id": int(cls_id),
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return detections
