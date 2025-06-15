"""Draw detections and the per-seat occupancy verdict on a cabin frame."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .data.classes import DETECTION_CLASSES, OCCUPANCY_CLASSES, SEAT_POSITIONS
from .models.base import Prediction

COLORS = {
    "infant_seat": (230, 80, 80),
    "child_seat": (240, 160, 40),
    "person": (60, 170, 240),
    "everyday_object": (120, 200, 120),
}


def draw(
    image: Image.Image,
    pred: Prediction,
    seat_regions: Optional[Sequence[np.ndarray]] = None,
    seat_labels: Optional[Sequence[int]] = None,
    score_thresh: float = 0.4,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    d = ImageDraw.Draw(canvas)
    if seat_regions is not None:
        for pos, region, label in zip(SEAT_POSITIONS, seat_regions, seat_labels or [None] * len(SEAT_POSITIONS)):
            x1, y1, x2, y2 = [float(v) for v in region]
            d.rectangle([x1, y1, x2, y2], outline=(200, 200, 200), width=1)
            text = pos if label is None else f"{pos}: {OCCUPANCY_CLASSES[label]}"
            d.text((x1 + 3, y2 - 14), text, fill=(255, 255, 255))
    p = pred.filter(score_thresh)
    for box, score, label in zip(p.boxes, p.scores, p.labels):
        name = DETECTION_CLASSES[int(label)]
        color = COLORS.get(name, (255, 255, 0))
        d.rectangle([float(v) for v in box], outline=color, width=3)
        d.text((float(box[0]) + 3, float(box[1]) + 2), f"{name} {score:.2f}", fill=color)
    return canvas


def save(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
