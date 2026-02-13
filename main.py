from __future__ import annotations

import logging
import warnings

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

from pathlib import Path

from PIL import Image
import streamlit as st
import torch

from app.config import (
    DETECTRON_DEFAULT_WEIGHTS,
    FRCNN_FILENAME,
    FRCNN_TRAINING_ATTRS,
    FRCNN_METRICS_CACHE,
    HF_REPO_ID,
    METRICS_FILENAME_FRCNN,
    METRICS_FILENAME_YOLO,
    METRICS_REPO_ID,
    METRICS_REPO_TYPE,
    NEW_NAMES,
    YOLO_DEFAULT_WEIGHTS,
    YOLO_FILENAME,
    YOLO_METRICS_CACHE,
    YOLO_TRAINING_ATTRS,
)
from app.detectron import load_detectron_model, predict_detectron
from app.formatting import dict_to_rows_str
from app.hf import ensure_file_from_hf, maybe_download_weights
from app.metrics import load_metrics_cache
from app.ui import draw_detections, top_detection_summary
from app.yolo import load_yolo_model, predict_yolo

MODEL_OPTIONS = ("Faster R-CNN", "YOLO")


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


@st.cache_resource(show_spinner=False)
def _get_detectron_model(weights_path: str):
    return load_detectron_model(weights_path)


@st.cache_resource(show_spinner=False)
def _get_yolo_model(weights_path: str):
    return load_yolo_model(weights_path)


@st.cache_data(show_spinner=False)
def _load_metrics_cached(
    path_str: str,
    repo_id: str,
    filename: str,
    repo_type: str,
) -> dict | None:
    local_path = Path(path_str)
    try:
        ensure_file_from_hf(local_path, repo_id, filename, repo_type=repo_type)
    except Exception:
        pass
    return load_metrics_cache(local_path)


def _resolve_weights(model_choice: str) -> Path:
    if model_choice == "Faster R-CNN":
        return maybe_download_weights(
            weights_path=Path(DETECTRON_DEFAULT_WEIGHTS),
            repo_id=HF_REPO_ID,
            filename=FRCNN_FILENAME,
            model_label="Faster R-CNN",
        )
    return maybe_download_weights(
        weights_path=Path(YOLO_DEFAULT_WEIGHTS),
        repo_id=HF_REPO_ID,
        filename=YOLO_FILENAME,
        model_label="YOLO",
    )


def _run_inference(model_choice: str, image: Image.Image) -> list[dict]:
    if model_choice == "Faster R-CNN":
        weights = _resolve_weights(model_choice)
        predictor = _get_detectron_model(str(weights))
        with torch.inference_mode():
            return predict_detectron(predictor, image, NEW_NAMES)
    weights = _resolve_weights(model_choice)
    model = _get_yolo_model(str(weights))
    with torch.inference_mode():
        return predict_yolo(model, image, [])


def _render_predict_tab() -> None:
    model_choice = st.selectbox("Model", list(MODEL_OPTIONS))
    uploaded_file = st.file_uploader(
        "Upload X-ray image", type=["png", "jpg", "jpeg"]
    )
    image = Image.open(uploaded_file).convert("RGB") if uploaded_file else None

    if st.button("Predict", type="primary"):
        if image is None:
            st.error("Upload an image before running predictions.")
            return
        try:
            with st.spinner("Running inference..."):
                detections = _run_inference(model_choice, image)
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


def _render_models_tab() -> None:
    info_choice = st.selectbox("Model", list(MODEL_OPTIONS), key="info_model")
    if info_choice == "Faster R-CNN":
        st.subheader("Faster R-CNN (Detectron2)")
        st.markdown("Training attributes")
        st.table(dict_to_rows_str(FRCNN_TRAINING_ATTRS, "Attribute", "Value"))

        st.markdown("Metrics")
        cache = _load_metrics_cached(
            FRCNN_METRICS_CACHE,
            METRICS_REPO_ID,
            METRICS_FILENAME_FRCNN,
            METRICS_REPO_TYPE,
        )
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
        cache = _load_metrics_cached(
            YOLO_METRICS_CACHE,
            METRICS_REPO_ID,
            METRICS_FILENAME_YOLO,
            METRICS_REPO_TYPE,
        )
        if cache and "metrics" in cache:
            metrics = cache["metrics"]
            froc_curve = cache.get("froc_curve", [])
            _render_key_metrics(metrics, froc_curve)


def main() -> None:
    st.set_page_config(page_title="Bone X-Ray Detector")
    st.title("Bone X-Ray Detector")
    st.caption("Upload an X-ray image, choose a model, and run predictions.")

    tab_predict, tab_models = st.tabs(["Predict", "Models"])

    with tab_predict:
        _render_predict_tab()

    with tab_models:
        _render_models_tab()


if __name__ == "__main__":
    main()
