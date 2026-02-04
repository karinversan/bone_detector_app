# Bone X‑Ray Fracture Detector

Приложение для детекции переломов на рентген‑снимках: Streamlit UI + два детектора (Detectron2 Faster R‑CNN и Ultralytics YOLO) с фоновой оценкой метрик через Celery/Redis.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-%3E%3D1.53.1-FF4B4B" />
  <img alt="Detectron2" src="https://img.shields.io/badge/Detectron2-git-blueviolet" />
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-%3E%3D8.1.0-black" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C" />
  <img alt="Torchvision" src="https://img.shields.io/badge/Torchvision-0.16.2-EE4C2C" />
  <img alt="Celery" src="https://img.shields.io/badge/Celery-%3E%3D5.3-3E9F8C" />
  <img alt="Redis" src="https://img.shields.io/badge/Redis-7-DC382D" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED" />
</p>

---

## Highlights

- Один UI для двух моделей: Faster R‑CNN (Detectron2) и YOLO (Ultralytics).
- Асинхронный расчёт метрик через Celery + Redis без блокировки UI.
- Кеш метрик в `cache/` и повторное отображение без пересчёта.
- Единый 6‑классовый протокол оценки (merge `humerus fracture → humerus`) для сравнения моделей.
- Метрики: COCO AP + FROC/Sensitivity@FP‑per‑image.
- Запуск всего стека через Docker Compose.

---

## Demo

<details>
  <summary><b>Demo (gif/video)</b></summary>
  <p>Файл добавлю позже</p>
</details>

---

## Architecture (словами)

Поток приложения: Streamlit UI принимает изображение → базовый preprocessing (PIL‑RGB, BGR для Detectron2) → модель делает предсказания → постобработка/визуализация боксов → метрики считаются в фоне через Celery и кешируются в `cache/`.

---

## Данные и загрузка

**Источник:** Kaggle dataset `pkdarabi/bone-fracture-detection-computer-vision-project`  
**Автор:** pkdarabi (Kaggle)  
**Лицензия:** CC BY 4.0 (указана в `data/BoneFractureYolo8/data.yaml`)  
**Описание:** набор рентген‑снимков верхней конечности с разметкой переломов по классам; аннотации представлены как bounding boxes или сегментационные маски.  
**Классы (7):** Elbow Positive, Fingers Positive, Forearm Fracture, Humerus Fracture, Humerus, Shoulder Fracture, Wrist Positive.  
**Особенности:** датасет уже содержит аугментации (повороты, изменения яркости/контраста). В выборке есть снимки без переломов (пустые label‑файлы).

**Как скачать и положить в нужную папку** (нужно для конвертации и подсчёта метрик):

```bash
pip install kaggle

# положите kaggle.json в ~/.kaggle/ (доступ к Kaggle API)
kaggle datasets download -d pkdarabi/bone-fracture-detection-computer-vision-project -p data --unzip
```

Ожидаемая структура (используется в `app/config.py` и для метрик):
```
data/BoneFractureYolo8/
  train/images, train/labels
  valid/images, valid/labels
  test/images,  test/labels
```
Если папка после скачивания называется иначе — переименуйте в `BoneFractureYolo8`.

Для цитирования датасета указана DOI: `10.13140/RG.2.2.14400.34569`  
ResearchGate: `https://www.researchgate.net/publication/382268240_Bone_Fracture_Detection_Computer_Vision_Project`

---

## Как я решал задачу

- Проанализировала разметку YOLO‑seg и распределение классов.
- Объединила `humerus fracture` с `humerus`, т.к. в исходных label‑файлах всего 3 объекта этого класса.
- Сделала единый протокол сравнения моделей через COCO‑JSON и маппинг классов.
- Сконструировала общий инференс‑поток и унифицированный формат детекций для отрисовки.
- Вынесла расчёт метрик в Celery‑очередь и добавила кеш + lock‑механику.
- Упаковала приложение, воркер и Redis в Docker Compose.
- Обучение зафиксировано в исследовательском ноутбуке:
  - YOLO: 50 эпох, `imgsz=640`.
  - Faster R‑CNN: `NUM_CLASSES=6`, `imgsz=1024`, `FILTER_EMPTY_ANNOTATIONS=False` (включая снимки без переломов).
- В обучение включены все данные, включая изображения без переломов.
- Датасет уже содержит аугментации; дополнительная аугментация улучшений не дала.

---

## Проблемы и решения

- Разная схема классов (7 vs 6) → мёрж и маппинг классов → сопоставимые метрики.
- Формат YOLO‑seg разметки → конвертация в COCO → корректный COCOeval.
- Detectron2 ожидает BGR → явная конвертация RGB→BGR → корректный инференс.
- Долгий пересчёт метрик → Celery + Redis + кеш → UI остаётся отзывчивым.

---

## Результаты

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

Интерпретация: на этом протоколе Faster R‑CNN показывает более высокую Sens@1 FP/image при одинаковых условиях оценки.

---

## Запуск

### Docker Compose
```bash
docker compose up --build
```
Открыть: `http://localhost:8501`

### Локально
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
redis-server
.venv/bin/celery -A app.celery_app worker --loglevel=info
.venv/bin/streamlit run main.py
```

---

## Ограничения и что дальше
- Метрики валидны для текущего сплита; внешней валидации нет.
- Результаты зависят от качества разметки и источника снимков.
- Данные не хранятся в репозитории и должны быть скачаны отдельно (см. раздел «Данные и загрузка»).
- В репозитории нет клинической или регуляторной валидации; проект не является медицинским устройством и предназначен для research/education.
