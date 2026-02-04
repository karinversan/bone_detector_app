from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image

from .config import (
    MERGE_FROM_OLD,
    MERGE_TO_OLD,
    NEW_NAMES,
    OLD_NAMES,
    OLD_TO_NEW,
)


def yolo_seg_to_coco_merged(
    images_dir: Path, labels_dir: Path, out_json: Path
) -> int:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NEW_NAMES)],
    }

    ann_id = 1
    img_id = 1

    img_paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        img_paths += list(images_dir.rglob(ext))
    img_paths = sorted(img_paths)

    bad_class_lines = 0

    for img_path in img_paths:
        with Image.open(img_path) as im:
            width, height = im.size

        coco["images"].append(
            {
                "id": img_id,
                "file_name": str(img_path.relative_to(images_dir)),
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{img_path.stem}.txt"
        lines = []
        if label_path.exists():
            lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 7:
                continue

            cls_old = int(float(parts[0]))
            if not (0 <= cls_old < len(OLD_NAMES)):
                bad_class_lines += 1
                continue

            if cls_old == MERGE_FROM_OLD:
                cls_old = MERGE_TO_OLD
            if cls_old not in OLD_TO_NEW:
                continue
            cls_new = OLD_TO_NEW[cls_old]

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            xs_px = [max(0.0, min(width - 1.0, x * width)) for x in xs]
            ys_px = [max(0.0, min(height - 1.0, y * height)) for y in ys]

            xmin, xmax = min(xs_px), max(xs_px)
            ymin, ymax = min(ys_px), max(ys_px)

            bw = max(1.0, xmax - xmin)
            bh = max(1.0, ymax - ymin)

            bbox = [float(xmin), float(ymin), float(bw), float(bh)]

            poly: list[float] = []
            for x, y in zip(xs_px, ys_px):
                poly += [float(x), float(y)]
            seg = [poly] if len(poly) >= 6 else []

            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_new + 1,
                    "bbox": bbox,
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "segmentation": seg,
                }
            )
            ann_id += 1

        img_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(coco, ensure_ascii=False))

    return bad_class_lines
