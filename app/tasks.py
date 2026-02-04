from __future__ import annotations

import logging
import warnings
from pathlib import Path

from .celery_app import celery_app


@celery_app.task(name="app.compute_metrics")
def compute_metrics() -> str:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release.*",
    )
    logging.getLogger("torch.fx").setLevel(logging.ERROR)
    logging.getLogger("torch.fx._symbolic_trace").setLevel(logging.ERROR)

    from .config import (
        DETECTRON_DEFAULT_WEIGHTS,
        FRCNN_FILENAME,
        FRCNN_METRICS_CACHE,
        FRCNN_VAL_IMAGES,
        FRCNN_VAL_JSON,
        FRCNN_VAL_LABELS,
        HF_REPO_ID,
        METRICS_LOCK,
        YOLO_DEFAULT_WEIGHTS,
        YOLO_FILENAME,
        YOLO_METRICS_CACHE,
        YOLO_VAL_IMAGES,
        YOLO_VAL_JSON,
        YOLO_VAL_LABELS,
    )
    from .metrics import (
        get_or_compute_frcnn_metrics_with_download,
        get_or_compute_yolo_metrics_with_download,
    )

    lock_path = Path(METRICS_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        get_or_compute_frcnn_metrics_with_download(
            HF_REPO_ID,
            Path(DETECTRON_DEFAULT_WEIGHTS),
            FRCNN_FILENAME,
            Path(FRCNN_VAL_JSON),
            Path(FRCNN_VAL_IMAGES),
            Path(FRCNN_VAL_LABELS),
            Path(FRCNN_METRICS_CACHE),
        )
        get_or_compute_yolo_metrics_with_download(
            HF_REPO_ID,
            Path(YOLO_DEFAULT_WEIGHTS),
            YOLO_FILENAME,
            Path(YOLO_VAL_JSON),
            Path(YOLO_VAL_IMAGES),
            Path(YOLO_VAL_LABELS),
            Path(YOLO_METRICS_CACHE),
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()

    return "ok"
