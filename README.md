# Bone X‑Ray Fracture Detector

An application for fracture detection on X‑ray images: Streamlit UI + two detectors (Detectron2 Faster R‑CNN and Ultralytics YOLO) with background metric computation via Celery/Redis.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-%3E%3D1.53.1-FF4B4B" />
  <img alt="Detectron2" src="https://img.shields.io/badge/Detectron2-git-blueviolet" />
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-%3E%3D8.1.0-black" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C" />
  <img alt="Celery" src="https://img.shields.io/badge/Celery-%3E%3D5.3-3E9F8C" />
  <img alt="Redis" src="https://img.shields.io/badge/Redis-7-DC382D" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED" />
</p>

---

## Highlights

- One UI for two models: Faster R‑CNN (Detectron2) and YOLO (Ultralytics).
- Asynchronous metric computation via Celery + Redis without blocking the UI.
- Metrics cache in `cache/` with instant reuse.
- Unified 6‑class evaluation protocol (merge `humerus fracture → humerus`) for model comparison.
- Metrics: COCO AP + FROC/Sensitivity@FP‑per‑image.
- Full stack runs via Docker Compose.

---

## Demo


<summary><b>Demo (gif/video)</b></summary>
<p><img src="media/demo.gif" alt="Demo" /></p>


---

## Architecture

Flow: Streamlit UI accepts an image → basic preprocessing (PIL‑RGB, BGR for Detectron2) → model inference → post‑processing/box visualization → metrics are computed in the background via Celery and cached in `cache/`.

---

## How I approached the task

- Analyzed YOLO‑seg labels and class distribution.
- Merged `humerus fracture` into `humerus` because that class has only 3 objects in the original labels.
- Built a unified evaluation protocol via COCO‑JSON and class mapping.
- Implemented a shared inference flow and a unified detection format for visualization.
- Moved metric computation to a Celery queue and added cache + lock mechanism.
- Packaged the app, worker, and Redis in Docker Compose.
- Training was tracked in a research notebook:
  - YOLO: 50 epochs, `imgsz=640`.
  - Faster R‑CNN: `NUM_CLASSES=6`, `imgsz=1024`, `FILTER_EMPTY_ANNOTATIONS=False` (includes images without fractures).
- All data, including images without fractures, were used for training.
- The dataset already contains augmentations; additional augmentation did not improve results.

---

## Problems and solutions

- Different class schemes (7 vs 6) → class merge + mapping → comparable metrics.
- YOLO‑seg labels → COCO conversion → correct COCOeval.
- Detectron2 expects BGR → explicit RGB→BGR conversion → correct inference.
- Long metric computation → Celery + Redis + cache → responsive UI.

---

## Results (current split in this repo, protocol: 6‑class merged)

<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">AP50‑95</th>
      <th align="center">AP50</th>
      <th align="center">Sens@1 FP/image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Faster R‑CNN</b></td>
      <td align="center"><b>0.0881</b></td>
      <td align="center"><b>0.2707</b></td>
      <td align="center"><b>0.4706</b></td>
    </tr>
    <tr>
      <td><b>YOLO</b> (6‑class)</td>
      <td align="center"><b>0.0584</b></td>
      <td align="center"><b>0.1602</b></td>
      <td align="center"><b>0.2353</b></td>
    </tr>
  </tbody>
</table>

Interpretation: under this protocol, Faster R‑CNN shows higher Sens@1 FP/image at the same evaluation settings.

---

## Run

### Docker Compose
```bash
docker compose up --build
```
Open: `http://localhost:8501`

### Local
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
redis-server
.venv/bin/celery -A app.celery_app worker --loglevel=info
.venv/bin/streamlit run main.py
```

---


## Data and download

**Source:** Kaggle dataset `pkdarabi/bone-fracture-detection-computer-vision-project`  
**Author:** pkdarabi (Kaggle)  
**License:** CC BY 4.0 (listed in `data/BoneFractureYolo8/data.yaml`)  
**Description:** a set of upper‑extremity X‑ray images with fracture annotations by class; annotations are provided as bounding boxes or pixel‑level segmentation masks.  
**Classes (7):** Elbow Positive, Fingers Positive, Forearm Fracture, Humerus Fracture, Humerus, Shoulder Fracture, Wrist Positive.  
**Notes:** the dataset already includes augmentations (rotations, brightness/contrast changes). The split contains images without fractures (empty label files).

**Download and place in the expected folder** (required for conversion and metric computation):

```bash
pip install kaggle

# place kaggle.json into ~/.kaggle/ (Kaggle API access)
kaggle datasets download -d pkdarabi/bone-fracture-detection-computer-vision-project -p data --unzip
```

Expected structure (used in `app/config.py` and for metrics):
```
data/BoneFractureYolo8/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
```
If the extracted folder has a different name, rename it to `BoneFractureYolo8`.

Dataset citation DOI: `10.13140/RG.2.2.14400.34569`  
ResearchGate: `https://www.researchgate.net/publication/382268240_Bone_Fracture_Detection_Computer_Vision_Project`

---

## Limitations and next steps

- Metrics are valid only for the current split; no external validation.
- Training is documented in a notebook, not a fully reproducible pipeline/script.
- Results depend on label quality and imaging source.
- Data are not stored in the repo and must be downloaded separately (see “Data and download”).
- No clinical/regulatory validation; this is not a medical device and is intended for research/education.
