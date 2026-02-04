from __future__ import annotations

import json
from pathlib import Path
import time

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


def _normalize_detectron_metrics(results: dict) -> dict:
    res = results.get("bbox", results)
    mapping = {
        "AP50-95": res.get("AP"),
        "AP50": res.get("AP50"),
        "AP75": res.get("AP75"),
        "AP-small": res.get("APs"),
        "AP-medium": res.get("APm"),
        "AP-large": res.get("APl"),
        "AR@1": res.get("AR@1"),
        "AR@10": res.get("AR@10"),
        "AR@100": res.get("AR@100"),
        "AR-small": res.get("ARs"),
        "AR-medium": res.get("ARm"),
        "AR-large": res.get("ARl"),
    }
    return {k: v for k, v in mapping.items() if v is not None}

def _per_class_ap(coco_eval, class_names: list[str]) -> dict:
    import numpy as np

    precision = coco_eval.eval.get("precision")
    if precision is None:
        return {}
    # precision: [T, R, K, A, M]
    prec = precision[:, :, :, 0, -1]
    per_class = {}
    for k, name in enumerate(class_names):
        pk = prec[:, :, k]
        valid = pk[pk > -1]
        per_class[name] = float(np.mean(valid)) if valid.size else None
    return per_class


def _f1_at_iou50(coco_eval) -> float | None:
    import numpy as np

    ious = coco_eval.params.iouThrs
    t = int(np.where(np.isclose(ious, 0.5))[0][0])
    precision = coco_eval.eval.get("precision")
    recall = coco_eval.eval.get("recall")
    if precision is None or recall is None:
        return None
    p = precision[t, :, :, 0, -1]
    p = p[p > -1]
    r = recall[t, :, 0, -1]
    r = r[r > -1]
    if p.size == 0 or r.size == 0:
        return None
    p_mean = float(np.mean(p))
    r_mean = float(np.mean(r))
    if p_mean + r_mean == 0:
        return None
    return (2 * p_mean * r_mean) / (p_mean + r_mean)


def _stats_to_coco_metrics(stats: list[float]) -> dict:
    if len(stats) < 12:
        return {}
    return {
        "AP50-95": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP-small": stats[3],
        "AP-medium": stats[4],
        "AP-large": stats[5],
        "AR@1": stats[6],
        "AR@10": stats[7],
        "AR@100": stats[8],
        "AR-small": stats[9],
        "AR-medium": stats[10],
        "AR-large": stats[11],
    }


def compute_detectron_metrics(
    weights_path: Path, coco_json: Path, image_root: Path, labels_dir: Path | None
) -> tuple[dict, dict, int]:
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
    per_class = {}
    if isinstance(results, dict):
        metrics = _normalize_detectron_metrics(results)
    if hasattr(evaluator, "_coco_eval") and evaluator._coco_eval is not None:
        coco_eval = evaluator._coco_eval
        f1 = _f1_at_iou50(coco_eval)
        if f1 is not None:
            metrics["F1@0.5"] = f1
        per_class = _per_class_ap(coco_eval, NEW_NAMES)
    return metrics, per_class, bad_class_lines


def compute_yolo_metrics(
    weights_path: Path, coco_json: Path, image_root: Path
) -> tuple[dict, dict]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    from PIL import Image

    if not coco_json.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_json}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    model = load_yolo_model(str(weights_path))
    coco_gt = COCO(str(coco_json))
    img_ids = coco_gt.getImgIds()
    preds = []

    for img_id in img_ids:
        info = coco_gt.loadImgs(img_id)[0]
        img_path = image_root / info["file_name"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            result = model.predict(source=np.array(im), verbose=False)[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            preds.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cls_id) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    coco_dt = coco_gt.loadRes(preds) if preds else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    metrics = _stats_to_coco_metrics(list(coco_eval.stats))
    f1 = _f1_at_iou50(coco_eval)
    if f1 is not None:
        metrics["F1@0.5"] = f1
    per_class = _per_class_ap(coco_eval, NEW_NAMES)
    return metrics, per_class


def _to_jsonable(value: object):
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return str(value)


def _serialize_metrics(metrics: dict) -> dict:
    return {str(k): _to_jsonable(v) for k, v in metrics.items()}


def _load_metrics_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_metrics_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def get_or_compute_frcnn_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
    cache_path: Path,
) -> tuple[dict, dict, int, bool]:
    cache = _load_metrics_cache(cache_path)
    if cache and "metrics" in cache:
        return (
            cache["metrics"],
            cache.get("per_class_ap", {}),
            int(cache.get("bad_class_lines", 0)),
            True,
        )

    weights = maybe_download_weights(weights_path, repo_id, filename, "Faster R-CNN")
    ensure_weights_exist(weights, "Faster R-CNN")
    metrics, per_class, bad_class_lines = compute_detectron_metrics(
        weights, coco_json, image_root, labels_dir
    )
    payload = {
        "metrics": _serialize_metrics(metrics),
        "per_class_ap": _serialize_metrics(per_class),
        "bad_class_lines": int(bad_class_lines),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_metrics_cache(cache_path, payload)
    return (
        payload["metrics"],
        payload["per_class_ap"],
        payload["bad_class_lines"],
        False,
    )


def get_or_compute_yolo_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    cache_path: Path,
) -> tuple[dict, dict, bool]:
    cache = _load_metrics_cache(cache_path)
    if cache and "metrics" in cache:
        return cache["metrics"], cache.get("per_class_ap", {}), True

    weights = maybe_download_weights(weights_path, repo_id, filename, "YOLO")
    ensure_weights_exist(weights, "YOLO")
    metrics, per_class = compute_yolo_metrics(weights, coco_json, image_root)
    payload = {
        "metrics": _serialize_metrics(metrics),
        "per_class_ap": _serialize_metrics(per_class),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_metrics_cache(cache_path, payload)
    return payload["metrics"], payload["per_class_ap"], False


def compute_frcnn_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
) -> tuple[dict, dict, int]:
    weights = maybe_download_weights(weights_path, repo_id, filename, "Faster R-CNN")
    ensure_weights_exist(weights, "Faster R-CNN")
    return compute_detectron_metrics(weights, coco_json, image_root, labels_dir)


def compute_yolo_metrics_with_download(
    repo_id: str, weights_path: Path, filename: str, coco_json: Path, image_root: Path
) -> tuple[dict, dict]:
    weights = maybe_download_weights(weights_path, repo_id, filename, "YOLO")
    ensure_weights_exist(weights, "YOLO")
    return compute_yolo_metrics(weights, coco_json, image_root)
