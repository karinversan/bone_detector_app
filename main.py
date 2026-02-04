from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
import torch


WEIGHTS_DIR = "weights"
FRCNN_FILENAME = os.getenv("HF_FILENAME_FRCNN", "model_final_frcnn.pth")
YOLO_FILENAME = os.getenv("HF_FILENAME_YOLO", "best_yolo26m_640.pt")
DETECTRON_DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/{FRCNN_FILENAME}"
YOLO_DEFAULT_WEIGHTS = f"{WEIGHTS_DIR}/{YOLO_FILENAME}"
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

NEW_NAMES = [
    "elbow positive",
    "fingers positive",
    "forearm fracture",
    "humerus",
    "shoulder fracture",
    "wrist positive",
]


def ensure_weights_exist(weights_path: Path, model_label: str) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{model_label} weights not found at: {weights_path}. "
            "Place the weights at the default path in the project."
        )


def maybe_download_weights(
    weights_path: Path, repo_id: str, filename: str, model_label: str
) -> Path:
    if weights_path.exists():
        return weights_path
    if not repo_id:
        raise FileNotFoundError(
            f"{model_label} weights not found at: {weights_path}. "
            "Set HF_REPO_ID_* env var or place the weights locally."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it to download weights."
        ) from exc

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=weights_path.parent,
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def draw_detections(image: Image.Image, detections: Iterable[dict]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
        label = det["label"]
        score = det.get("score")
        text = f"{label} {score:.2f}" if score is not None else label

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text_bbox = draw.textbbox((0, 0), text)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = x1
        text_y = max(0, y1 - text_h - 4)
        draw.rectangle(
            [text_x, text_y, text_x + text_w + 6, text_y + text_h + 4],
            fill="red",
        )
        draw.text((text_x + 3, text_y + 2), text, fill="white")
    return annotated


@st.cache_resource(show_spinner=False)
def load_detectron_model(weights_path: str):
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError as exc:
        raise RuntimeError(
            "Detectron2 is not installed. Install detectron2 to use the Faster R-CNN model."
        ) from exc
    

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6

    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [
        [0.33, 0.5, 1.0, 2.0, 3.0]
    ] * 5

    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = weights_path
    return DefaultPredictor(cfg)


def predict_detectron(
    predictor, image: Image.Image, class_names: list[str]
) -> list[dict]:
    image_bgr = np.array(image)[:, :, ::-1]
    outputs = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy().astype(int)

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        label = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        detections.append(
            {
                "label": label,
                "class_id": int(cls_id),
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return detections


@st.cache_resource(show_spinner=False)
def load_yolo_model(weights_path: str):
    try:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
        try:
            from torch.serialization import safe_globals
        except Exception:
            safe_globals = None
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install ultralytics to use the YOLO model."
        ) from exc
    if safe_globals is not None:
        with safe_globals([DetectionModel]):
            return YOLO(weights_path)
    try:
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
    return YOLO(weights_path)


def resolve_yolo_label(model, class_names: list[str], cls_id: int) -> str:
    if class_names and 0 <= cls_id < len(class_names):
        return class_names[cls_id]
    if hasattr(model, "names"):
        names = model.names
        if isinstance(names, dict) and cls_id in names:
            return str(names[cls_id])
        if isinstance(names, list) and 0 <= cls_id < len(names):
            return str(names[cls_id])
    return f"class_{cls_id}"


def predict_yolo(model, image: Image.Image, class_names: list[str]) -> list[dict]:
    results = model.predict(source=np.array(image), verbose=False)
    result = results[0]
    if result.boxes is None:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    detections = []
    for box, score, cls_id in zip(boxes, scores, classes):
        label = resolve_yolo_label(model, class_names, cls_id)
        detections.append(
            {
                "label": label,
                "class_id": int(cls_id),
                "score": float(score),
                "box": [float(v) for v in box],
            }
        )
    return detections


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
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = None

        if st.button("Predict", type="primary"):
            if image is None:
                st.error("Upload an image before running predictions.")
                return

            weights = Path(weights_path).expanduser()
            repo_id = (
                HF_REPO_ID if model_choice == "Faster R-CNN" else HF_REPO_ID
            )
            filename = (
                FRCNN_FILENAME if model_choice == "Faster R-CNN" else YOLO_FILENAME
            )
            try:
                weights = maybe_download_weights(
                    weights, repo_id, filename, model_choice
                )
                ensure_weights_exist(weights, model_choice)
            except FileNotFoundError as exc:
                st.error(str(exc))
                return

            with st.spinner("Running inference..."):
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

            if detections:
                st.info(detections[0]["label"])
            else:
                st.info("No fracture")
            st.image(annotated, caption="Predictions")

    with tab_models:
        st.subheader("Faster R-CNN (Detectron2)")
        st.markdown(
            f"- Weights: `{DETECTRON_DEFAULT_WEIGHTS}`\n"
            "- Config: `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`\n"
            "- Input size: 1024 (min/max)\n"
            f"- Classes ({len(NEW_NAMES)}): {', '.join(NEW_NAMES)}\n"
            f"- HF repo: `{HF_REPO_ID or 'not set'}`"
        )
        st.subheader("YOLO (Ultralytics)")
        st.markdown(
            f"- Weights: `{YOLO_DEFAULT_WEIGHTS}`\n"
            "- Classes: from checkpoint metadata\n"
            f"- HF repo: `{HF_REPO_ID or 'not set'}`"
        )


if __name__ == "__main__":
    main()
