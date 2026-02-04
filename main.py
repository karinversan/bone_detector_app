"""Streamlit entrypoint.

Modules:
- app.config: paths, labels, training attrs
- app.detectron: Detectron2 cfg and inference
- app.yolo: Ultralytics inference
- app.metrics: metrics computation
- app.hf: optional HF download
- app.ui: rendering helpers
- app.formatting: table/metric formatting
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import streamlit as st

from app.config import (
    DETECTRON_DEFAULT_WEIGHTS,
    FRCNN_FILENAME,
    FRCNN_TRAINING_ATTRS,
    FRCNN_VAL_IMAGES,
    FRCNN_VAL_JSON,
    FRCNN_VAL_LABELS,
    HF_REPO_ID,
    NEW_NAMES,
    YOLO_DATA_YAML,
    YOLO_DEFAULT_WEIGHTS,
    YOLO_FILENAME,
    YOLO_TRAINING_ATTRS,
)
from app.detectron import load_detectron_model, predict_detectron
from app.formatting import dict_to_rows, dict_to_rows_str, metrics_to_display, metrics_with_f1
from app.metrics import (
    compute_frcnn_metrics_with_download,
    compute_yolo_metrics_with_download,
)
from app.ui import draw_detections, top_detection_summary
from app.yolo import load_yolo_model, predict_yolo


def run_frcnn_metrics() -> None:
    try:
        metrics, bad_class_lines = compute_frcnn_metrics_with_download(
            HF_REPO_ID,
            Path(DETECTRON_DEFAULT_WEIGHTS),
            FRCNN_FILENAME,
            Path(FRCNN_VAL_JSON),
            Path(FRCNN_VAL_IMAGES),
            Path(FRCNN_VAL_LABELS),
        )
        st.session_state["metrics_frcnn"] = metrics
        st.session_state.pop("metrics_frcnn_error", None)
        if bad_class_lines:
            st.session_state[
                "metrics_frcnn_warning"
            ] = f"Skipped {bad_class_lines} label lines with invalid class_id."
        else:
            st.session_state.pop("metrics_frcnn_warning", None)
    except Exception as exc:
        st.session_state["metrics_frcnn_error"] = str(exc)


def run_yolo_metrics() -> None:
    try:
        metrics = compute_yolo_metrics_with_download(
            HF_REPO_ID,
            Path(YOLO_DEFAULT_WEIGHTS),
            YOLO_FILENAME,
            Path(YOLO_DATA_YAML),
        )
        st.session_state["metrics_yolo"] = metrics
        st.session_state.pop("metrics_yolo_error", None)
    except Exception as exc:
        st.session_state["metrics_yolo_error"] = str(exc)


def main() -> None:
    st.set_page_config(page_title="Bone X-Ray Detector")
    st.title("Bone X-Ray Detector")
    st.caption("Upload an X-ray image, choose a model, and run predictions.")

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
            st.markdown(
                f"- Weights: `{DETECTRON_DEFAULT_WEIGHTS}`\n"
                f"- HF repo: `{HF_REPO_ID or 'not set'}`\n"
                f"- Classes ({len(NEW_NAMES)}): {', '.join(NEW_NAMES)}\n"
                f"- Val images: `{FRCNN_VAL_IMAGES}`\n"
                f"- Val labels (YOLO seg): `{FRCNN_VAL_LABELS}`\n"
                f"- Val COCO json: `{FRCNN_VAL_JSON}`"
            )
            st.markdown("Training attributes")
            st.table(dict_to_rows_str(FRCNN_TRAINING_ATTRS, "Attribute", "Value"))

            st.markdown("Metrics")
            if st.button("Compute metrics", key="frcnn_metrics"):
                with st.spinner("Computing metrics..."):
                    run_frcnn_metrics()
            if "metrics_frcnn_warning" in st.session_state:
                st.warning(st.session_state["metrics_frcnn_warning"])
            if "metrics_frcnn_error" in st.session_state:
                st.error(f"Metrics failed: {st.session_state['metrics_frcnn_error']}")
            elif "metrics_frcnn" in st.session_state:
                metrics = metrics_with_f1(st.session_state["metrics_frcnn"])
                metrics_display = metrics_to_display(metrics)
                st.table(dict_to_rows(metrics_display, "Metric", "Value"))
            else:
                st.info("Metrics not computed yet.")
        else:
            st.subheader("YOLO (Ultralytics)")
            st.markdown(
                f"- Weights: `{YOLO_DEFAULT_WEIGHTS}`\n"
                f"- HF repo: `{HF_REPO_ID or 'not set'}`\n"
                f"- Data yaml: `{YOLO_DATA_YAML}`\n"
                "- Classes: from checkpoint metadata"
            )
            st.markdown("Training attributes")
            st.table(dict_to_rows_str(YOLO_TRAINING_ATTRS, "Attribute", "Value"))

            st.markdown("Metrics")
            if st.button("Compute metrics", key="yolo_metrics"):
                with st.spinner("Computing metrics..."):
                    run_yolo_metrics()
            if "metrics_yolo_error" in st.session_state:
                st.error(f"Metrics failed: {st.session_state['metrics_yolo_error']}")
            elif "metrics_yolo" in st.session_state:
                metrics = metrics_with_f1(st.session_state["metrics_yolo"])
                metrics_display = metrics_to_display(metrics)
                st.table(dict_to_rows(metrics_display, "Metric", "Value"))
            else:
                st.info("Metrics not computed yet.")


if __name__ == "__main__":
    main()
