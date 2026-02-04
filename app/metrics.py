from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import time

from .datasets import yolo_seg_to_coco_merged
from .detectron import build_detectron_cfg
from .hf import ensure_weights_exist, maybe_download_weights
from .yolo import load_yolo_model
from .config import (
    FROC_FP_PER_IMAGE,
    FROC_IOU,
    MERGE_FROM_OLD,
    MERGE_TO_OLD,
    METRICS_VERSION,
    NEW_NAMES,
    OLD_TO_NEW,
)


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


def _quiet_call(fn):
    buffer = StringIO()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        return fn()


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


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _compute_froc(
    coco_gt, preds: list[dict], iou_thr: float, fp_per_image_targets: list[float]
) -> tuple[dict, list[dict]]:
    from collections import defaultdict

    gts = defaultdict(list)
    ann_ids = coco_gt.getAnnIds()
    anns = coco_gt.loadAnns(ann_ids)
    for ann in anns:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        gts[(ann["image_id"], ann["category_id"])].append(
            [x, y, x + w, y + h]
        )

    dets = defaultdict(list)
    for pred in preds:
        x, y, w, h = pred["bbox"]
        dets[(pred["image_id"], pred["category_id"])].append(
            (pred["score"], [x, y, x + w, y + h])
        )

    tp_fp = []
    total_gt = sum(len(v) for v in gts.values())
    if total_gt == 0:
        return {f"Sensitivity@{fp} FP/image": 0.0 for fp in fp_per_image_targets}, []

    for key, det_list in dets.items():
        det_list.sort(key=lambda x: x[0], reverse=True)
        gt_list = gts.get(key, [])
        matched = [False] * len(gt_list)
        for score, bbox in det_list:
            best_iou = 0.0
            best_idx = -1
            for i, gt in enumerate(gt_list):
                if matched[i]:
                    continue
                iou = _compute_iou(bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thr and best_idx >= 0:
                matched[best_idx] = True
                tp_fp.append((score, 1))
            else:
                tp_fp.append((score, 0))

    tp_fp.sort(key=lambda x: x[0], reverse=True)
    num_images = len(coco_gt.getImgIds())
    cum_tp = 0
    cum_fp = 0
    curve = []
    for _, is_tp in tp_fp:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        fpi = cum_fp / max(num_images, 1)
        sens = cum_tp / total_gt
        curve.append({"fp_per_image": fpi, "sensitivity": sens})

    sens_at_fp = {}
    for target in fp_per_image_targets:
        best = 0.0
        for point in curve:
            if point["fp_per_image"] <= target:
                if point["sensitivity"] > best:
                    best = point["sensitivity"]
        sens_at_fp[f"Sensitivity@{target} FP/image"] = best

    return sens_at_fp, curve


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


def _predict_detectron_coco(weights_path: Path, coco_gt, image_root: Path) -> list[dict]:
    import numpy as np
    from PIL import Image

    try:
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed. Install detectron2 to compute metrics."
        ) from exc

    cfg = build_detectron_cfg(str(weights_path))
    predictor = DefaultPredictor(cfg)
    preds = []

    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        img_path = image_root / info["file_name"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image_bgr = np.array(im)[:, :, ::-1]
        outputs = predictor(image_bgr)
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy().astype(int)
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
    return preds


def compute_detectron_metrics(
    weights_path: Path, coco_json: Path, image_root: Path, labels_dir: Path | None
) -> tuple[dict, dict, list[dict], int]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

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

    coco_gt = _quiet_call(lambda: COCO(str(coco_json)))
    preds = _predict_detectron_coco(weights_path, coco_gt, image_root)
    coco_dt = (
        _quiet_call(lambda: coco_gt.loadRes(preds))
        if preds
        else _quiet_call(lambda: coco_gt.loadRes([]))
    )
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    _quiet_call(coco_eval.summarize)

    metrics = _stats_to_coco_metrics(list(coco_eval.stats))
    f1 = _f1_at_iou50(coco_eval)
    if f1 is not None:
        metrics["F1@0.5"] = f1
    per_class = _per_class_ap(coco_eval, NEW_NAMES)
    sens_at_fp, froc_curve = _compute_froc(
        coco_gt, preds, FROC_IOU, FROC_FP_PER_IMAGE
    )
    metrics.update(sens_at_fp)
    return metrics, per_class, froc_curve, bad_class_lines


def _map_yolo_class_to_merged(cls_id: int) -> int | None:
    if cls_id == MERGE_FROM_OLD:
        cls_id = MERGE_TO_OLD
    if cls_id not in OLD_TO_NEW:
        return None
    return OLD_TO_NEW[cls_id]


def compute_yolo_metrics(
    weights_path: Path,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
) -> tuple[dict, dict, list[dict], int]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import numpy as np
    from PIL import Image

    if not image_root.exists():
        raise FileNotFoundError(f"Image folder not found: {image_root}")

    bad_class_lines = 0
    if coco_json.exists() and _coco_needs_rebuild(coco_json, len(NEW_NAMES)):
        coco_json.unlink()

    if not coco_json.exists():
        if labels_dir is None or not labels_dir.exists():
            raise FileNotFoundError(
                f"COCO json not found: {coco_json}. "
                "Provide labels_dir to build it from YOLO seg labels."
            )
        bad_class_lines = yolo_seg_to_coco_merged(image_root, labels_dir, coco_json)

    model = load_yolo_model(str(weights_path))
    coco_gt = _quiet_call(lambda: COCO(str(coco_json)))
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
            merged_id = _map_yolo_class_to_merged(int(cls_id))
            if merged_id is None:
                continue
            x1, y1, x2, y2 = box
            preds.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(merged_id) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    coco_dt = (
        _quiet_call(lambda: coco_gt.loadRes(preds))
        if preds
        else _quiet_call(lambda: coco_gt.loadRes([]))
    )
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    _quiet_call(coco_eval.summarize)
    metrics = _stats_to_coco_metrics(list(coco_eval.stats))
    f1 = _f1_at_iou50(coco_eval)
    if f1 is not None:
        metrics["F1@0.5"] = f1
    per_class = _per_class_ap(coco_eval, NEW_NAMES)
    sens_at_fp, froc_curve = _compute_froc(
        coco_gt, preds, FROC_IOU, FROC_FP_PER_IMAGE
    )
    metrics.update(sens_at_fp)
    return metrics, per_class, froc_curve, bad_class_lines


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


def _serialize_metrics(metrics):
    if isinstance(metrics, dict):
        return {str(k): _to_jsonable(v) for k, v in metrics.items()}
    if isinstance(metrics, list):
        return [_to_jsonable(v) for v in metrics]
    return _to_jsonable(metrics)


def _load_metrics_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_metrics_cache(path: Path, expected_version: int | None = METRICS_VERSION) -> dict | None:
    cache = _load_metrics_cache(path)
    if not cache:
        return None
    if expected_version is not None and cache.get("version") != expected_version:
        return None
    return cache


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
) -> tuple[dict, dict, list[dict], int, bool]:
    cache = _load_metrics_cache(cache_path)
    if cache and cache.get("version") == METRICS_VERSION and "metrics" in cache:
        return (
            cache["metrics"],
            cache.get("per_class_ap", {}),
            cache.get("froc_curve", []),
            int(cache.get("bad_class_lines", 0)),
            True,
        )

    weights = maybe_download_weights(weights_path, repo_id, filename, "Faster R-CNN")
    ensure_weights_exist(weights, "Faster R-CNN")
    metrics, per_class, froc_curve, bad_class_lines = compute_detectron_metrics(
        weights, coco_json, image_root, labels_dir
    )
    payload = {
        "version": METRICS_VERSION,
        "metrics": _serialize_metrics(metrics),
        "per_class_ap": _serialize_metrics(per_class),
        "froc_curve": _serialize_metrics(froc_curve),
        "bad_class_lines": int(bad_class_lines),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_metrics_cache(cache_path, payload)
    return (
        payload["metrics"],
        payload["per_class_ap"],
        payload["froc_curve"],
        payload["bad_class_lines"],
        False,
    )


def get_or_compute_yolo_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
    cache_path: Path,
) -> tuple[dict, dict, list[dict], bool]:
    cache = _load_metrics_cache(cache_path)
    if cache and cache.get("version") == METRICS_VERSION and "metrics" in cache:
        return (
            cache["metrics"],
            cache.get("per_class_ap", {}),
            cache.get("froc_curve", []),
            True,
        )

    weights = maybe_download_weights(weights_path, repo_id, filename, "YOLO")
    ensure_weights_exist(weights, "YOLO")
    metrics, per_class, froc_curve, bad_class_lines = compute_yolo_metrics(
        weights, coco_json, image_root, labels_dir
    )
    payload = {
        "version": METRICS_VERSION,
        "metrics": _serialize_metrics(metrics),
        "per_class_ap": _serialize_metrics(per_class),
        "froc_curve": _serialize_metrics(froc_curve),
        "bad_class_lines": int(bad_class_lines),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _save_metrics_cache(cache_path, payload)
    return payload["metrics"], payload["per_class_ap"], payload["froc_curve"], False


def compute_frcnn_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
) -> tuple[dict, dict, list[dict], int]:
    weights = maybe_download_weights(weights_path, repo_id, filename, "Faster R-CNN")
    ensure_weights_exist(weights, "Faster R-CNN")
    return compute_detectron_metrics(weights, coco_json, image_root, labels_dir)


def compute_yolo_metrics_with_download(
    repo_id: str,
    weights_path: Path,
    filename: str,
    coco_json: Path,
    image_root: Path,
    labels_dir: Path | None,
) -> tuple[dict, dict, list[dict], int]:
    weights = maybe_download_weights(weights_path, repo_id, filename, "YOLO")
    ensure_weights_exist(weights, "YOLO")
    return compute_yolo_metrics(weights, coco_json, image_root, labels_dir)
