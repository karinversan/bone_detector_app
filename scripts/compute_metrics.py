from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

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
    METRICS_FILENAME_FRCNN,
    METRICS_FILENAME_YOLO,
    METRICS_REPO_ID,
    METRICS_REPO_TYPE,
    METRICS_LOCK,
    YOLO_DEFAULT_WEIGHTS,
    YOLO_FILENAME,
    YOLO_METRICS_CACHE,
)
from app.hf import upload_file_to_hf
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
            Path(FRCNN_VAL_LABELS),
            Path(YOLO_METRICS_CACHE),
        )
        if METRICS_REPO_ID:
            upload_file_to_hf(
                METRICS_REPO_ID,
                METRICS_FILENAME_FRCNN,
                Path(FRCNN_METRICS_CACHE),
                repo_type=METRICS_REPO_TYPE,
                commit_message="Update FRCNN metrics",
            )
            upload_file_to_hf(
                METRICS_REPO_ID,
                METRICS_FILENAME_YOLO,
                Path(YOLO_METRICS_CACHE),
                repo_type=METRICS_REPO_TYPE,
                commit_message="Update YOLO metrics",
            )
    finally:
        if lock_path.exists():
            lock_path.unlink()


if __name__ == "__main__":
    main()
