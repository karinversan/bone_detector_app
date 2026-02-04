from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import logging
import warnings

from PIL import Image
import streamlit as st

from app.config import (
    DETECTRON_DEFAULT_WEIGHTS,
    FRCNN_TRAINING_ATTRS,
    FRCNN_METRICS_CACHE,
    METRICS_LOCK,
    NEW_NAMES,
    YOLO_DEFAULT_WEIGHTS,
    YOLO_METRICS_CACHE,
    YOLO_TRAINING_ATTRS,
)
from app.detectron import load_detectron_model, predict_detectron
from app.formatting import dict_to_rows_str
from app.metrics import load_metrics_cache
from app.ui import draw_detections, top_detection_summary
from app.yolo import load_yolo_model, predict_yolo


def _render_key_metrics(metrics: dict, froc_curve: list[dict]) -> None:
    def fmt(value: float | None) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "—"

    ap50_95 = metrics.get("AP50-95")
    ap50 = metrics.get("AP50")

    sens_items: list[tuple[float, float]] = []
    for key, value in metrics.items():
        if key.startswith("Sensitivity@"):
            try:
                fp = float(key.split("@", 1)[1].split()[0])
            except Exception:
                continue
            if isinstance(value, (int, float)):
                sens_items.append((fp, float(value)))
    sens_items.sort(key=lambda x: x[0])

    st.markdown("**Clinical focus: sensitivity (do not miss fractures)**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("AP50-95", fmt(ap50_95))
    with col2:
        st.metric("AP50", fmt(ap50))
    with col3:
        headline = sens_items[1] if len(sens_items) > 1 else (sens_items[0] if sens_items else None)
        if headline:
            st.metric(f"Sens@{headline[0]} FP/img", fmt(headline[1]))
        else:
            st.metric("Sensitivity", "—")

    if sens_items:
        rows = [{"FP/image": fp, "Sensitivity": sens} for fp, sens in sens_items]
        st.table(rows)

    if froc_curve:
        st.markdown("FROC curve (sensitivity vs FP/image)")
        try:
            import pandas as pd

            df = pd.DataFrame(froc_curve)
            st.line_chart(df.set_index("fp_per_image")["sensitivity"])
        except Exception:
            st.table(froc_curve[:50])


def _ensure_metrics_job() -> None:
    cache_ready = (
        Path(FRCNN_METRICS_CACHE).exists() and Path(YOLO_METRICS_CACHE).exists()
    )
    lock_path = Path(METRICS_LOCK)
    if cache_ready or lock_path.exists():
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("running")
    subprocess.Popen(
        [sys.executable, "scripts/compute_metrics.py"],
        cwd=Path(__file__).resolve().parent,
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
    )
    warnings.filterwarnings(
        "ignore",
        message="torch.meshgrid: in an upcoming release.*",
    )
    logging.getLogger("torch.fx").setLevel(logging.ERROR)

    st.set_page_config(page_title="Bone X-Ray Detector")
    st.title("Bone X-Ray Detector")
    st.caption("Upload an X-ray image, choose a model, and run predictions.")

    _ensure_metrics_job()

    tab_predict, tab_models = st.tabs(["Predict", "Models"])

    with tab_predict:
        model_choice = st.selectbox("Model", ["Faster R-CNN", "YOLO"])
        weights_path = (
            DETECTRON_DEFAULT_WEIGHTS
            if model_choice == "Faster R-CNN"
            else YOLO_DEFAULT_WEIGHTS
        )
        class_names = NEW_NAMES if model_choice == "Faster R-CNN" else []

        uploaded_file = st.file_uploader(
            "Upload X-ray image", type=["png", "jpg", "jpeg"]
        )
        image = Image.open(uploaded_file).convert("RGB") if uploaded_file else None

        if st.button("Predict", type="primary"):
            if image is None:
                st.error("Upload an image before running predictions.")
                return

            weights = Path(weights_path).expanduser()
            try:
                if model_choice == "Faster R-CNN":
                    predictor = load_detectron_model(str(weights))
                    detections = predict_detectron(predictor, image, class_names)
                else:
                    model = load_yolo_model(str(weights))
                    detections = predict_yolo(model, image, class_names)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

            annotated = draw_detections(image, detections)
            top_summary = top_detection_summary(detections)
            if top_summary:
                st.info(top_summary)
            else:
                st.info("No fracture")
            st.image(annotated, caption="Predictions")

    with tab_models:
        info_choice = st.selectbox("Model", ["Faster R-CNN", "YOLO"], key="info_model")
        if info_choice == "Faster R-CNN":
            st.subheader("Faster R-CNN (Detectron2)")
            st.markdown("Training attributes")
            st.table(dict_to_rows_str(FRCNN_TRAINING_ATTRS, "Attribute", "Value"))

            st.markdown("Metrics")
            cache = load_metrics_cache(Path(FRCNN_METRICS_CACHE))
            if cache and "metrics" in cache:
                metrics = cache["metrics"]
                froc_curve = cache.get("froc_curve", [])
                bad_class_lines = int(cache.get("bad_class_lines", 0))
                if bad_class_lines:
                    st.warning(
                        f"Skipped {bad_class_lines} label lines with invalid class_id."
                    )
                _render_key_metrics(metrics, froc_curve)
        else:
            st.subheader("YOLO (Ultralytics)")
            st.markdown("Training attributes")
            st.table(dict_to_rows_str(YOLO_TRAINING_ATTRS, "Attribute", "Value"))

            st.markdown("Metrics")
            cache = load_metrics_cache(Path(YOLO_METRICS_CACHE))
            if cache and "metrics" in cache:
                metrics = cache["metrics"]
                froc_curve = cache.get("froc_curve", [])
                _render_key_metrics(metrics, froc_curve)


if __name__ == "__main__":
    main()
