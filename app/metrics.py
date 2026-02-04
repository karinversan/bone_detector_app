from __future__ import annotations

import json
from pathlib import Path

from .datasets import yolo_seg_to_coco_merged
from .detectron import build_detectron_cfg
from .hf import ensure_weights_exist, maybe_download_weights
from .yolo import load_yolo_model
from .config import NEW_NAMES


def _coco_needs_rebuild(coco_json: Path, num_classes: int) -> bool:
    try:
        data = json.loads(coco_json.read_text())
    except Exception:
        return True

    cats = data.get("categories", [])
    cat_ids = {c.get("id") for c in cats if isinstance(c, dict)}
    if not cat_ids:
        return True
    if min(cat_ids) != 1 or max(cat_ids) != num_classes:
        return True

    ann_ids = set()
    for ann in data.get("annotations", [])[:1000]:
        if isinstance(ann, dict) and "category_id" in ann:
            ann_ids.add(ann["category_id"])
    if ann_ids and (min(ann_ids) < 1 or max(ann_ids) > num_classes):
        return True
    return False


def compute_detectron_metrics(
    weights_path: Path, coco_json: Path, image_root: Path, labels_dir: Path | None
) -> tuple[dict, int]:
    try:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data import build_detection_test_loader
        from detectron2.data.datasets import register_coco_instances
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.modeling import build_model
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed. Install detectron2 to compute metrics."
        ) from exc

    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    bad_class_lines = 0
    if coco_json.exists() and _coco_needs_rebuild(coco_json, len(NEW_NAMES)):
        coco_json.unlink()

    if not coco_json.exists():
        if labels_dir is None or not labels_dir.exists():
            raise FileNotFoundError(
                f"COCO json not found: {coco_json}. "
                "Provide FRCNN_VAL_LABELS to build it from YOLO seg labels."
            )
        bad_class_lines = yolo_seg_to_coco_merged(image_root, labels_dir, coco_json)

    dataset_name = "fracture_val_eval"
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, {}, str(coco_json), str(image_root))
        MetadataCatalog.get(dataset_name).set(thing_classes=NEW_NAMES)

    cfg = build_detectron_cfg(str(weights_path))
    cfg.DATASETS.TEST = (dataset_name,)

    model = build_model(cfg)
    DetectionCheckpointer(model).load(str(weights_path))
    model.eval()

    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="output_eval_frcnn")
    val_loader = build_detection_test_loader(cfg, dataset_name)
    results = inference_on_dataset(model, val_loader, evaluator)

    metrics = {}
    if isinstance(results, dict) and "bbox" in results:
        metrics.update(results["bbox"])
    elif isinstance(results, dict):
        metrics.update(results)
    return metrics, bad_class_lines


def compute_yolo_metrics(weights_path: Path, data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data yaml not found: {data_yaml}")
    model = load_yolo_model(str(weights_path))
    results = model.val(data=str(data_yaml), split="val", verbose=False)

    if hasattr(results, "results_dict"):
        return dict(results.results_dict)

    metrics = {}
    for name in ("map", "map50", "map75", "mp", "mr"):
        if hasattr(results, name):
            metrics[name] = getattr(results, name)
    if hasattr(results, "box"):
        box = results.box
        for name, key in (
            ("map", "map"),
            ("map50", "map50"),
            ("map75", "map75"),
            ("mp", "mp"),
            ("mr", "mr"),
        ):
            if hasattr(box, key) and name not in metrics:
                metrics[name] = getattr(box, key)
    return metrics


def compute_frcnn_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
) -> tuple[dict, int]:
    weights = maybe_download_weights(weights_path, repo_id, filename, "Faster R-CNN")
    ensure_weights_exist(weights, "Faster R-CNN")
    return compute_detectron_metrics(weights, coco_json, image_root, labels_dir)


def compute_yolo_metrics_with_download(
    repo_id: str, weights_path: Path, filename: str, data_yaml: Path
) -> dict:
    weights = maybe_download_weights(weights_path, repo_id, filename, "YOLO")
    ensure_weights_exist(weights, "YOLO")
    return compute_yolo_metrics(weights, data_yaml)
