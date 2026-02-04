from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import (
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
)
from app.metrics import (
    get_or_compute_frcnn_metrics_with_download,
    get_or_compute_yolo_metrics_with_download,
)


def main() -> None:
    lock_path = Path(METRICS_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("running")

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
            Path(FRCNN_VAL_JSON),
            Path(FRCNN_VAL_IMAGES),
            Path(YOLO_METRICS_CACHE),
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
