from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageDraw


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


def top_detection_summary(detections: list[dict]) -> str | None:
    if not detections:
        return None
    top = max(detections, key=lambda det: det.get("score", 0.0))
    label = top.get("label", "Detection")
    score = top.get("score")
    if isinstance(score, (int, float)):
        return f"{label} ({score:.2f})"
    return label
