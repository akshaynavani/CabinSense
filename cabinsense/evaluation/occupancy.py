"""Turn object detections into a per-seat occupancy verdict and score it.

For each seat region, every detection above ``score_thresh`` whose area lies at least
``min_overlap`` inside the region is a candidate; the seat takes the class of the candidate with
the highest ``score * overlap``. A seat with no candidates is ``empty``.

Besides plain seat accuracy we report *child presence* recall/precision (infant or child seat
occupied), which is the metric that matters for child-left-behind alerts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from ..data.classes import OCCUPANCY_CLASSES, SEAT_POSITIONS, detection_to_occupancy
from ..models.base import Prediction

EMPTY = 0
CHILD_CLASSES = {OCCUPANCY_CLASSES.index("infant_seat"), OCCUPANCY_CLASSES.index("child_seat")}


@dataclass
class SeatLayout:
    regions: Dict[str, Dict[str, List[float]]]
    score_thresh: float = 0.4
    min_overlap: float = 0.5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SeatLayout":
        cfg = yaml.safe_load(Path(path).read_text())
        regions = {"default": cfg["default"], **(cfg.get("vehicles") or {})}
        assign = cfg.get("assignment", {})
        return cls(regions, float(assign.get("score_thresh", 0.4)), float(assign.get("min_overlap", 0.5)))

    def seats_for(self, vehicle: Optional[str], width: int, height: int) -> List[np.ndarray]:
        layout = self.regions.get(vehicle or "", self.regions["default"])
        scale = np.array([width, height, width, height], dtype=np.float32)
        return [np.asarray(layout[pos], dtype=np.float32) * scale for pos in SEAT_POSITIONS]


def overlap_fraction(boxes: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Fraction of each box's area that lies inside ``region`` (both xyxy)."""
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(boxes[:, 0], region[0])
    iy1 = np.maximum(boxes[:, 1], region[1])
    ix2 = np.minimum(boxes[:, 2], region[2])
    iy2 = np.minimum(boxes[:, 3], region[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return np.where(area > 0, inter / np.maximum(area, 1e-6), 0.0)


def assign_seats(
    pred: Prediction,
    seat_regions: Sequence[np.ndarray],
    score_thresh: float = 0.4,
    min_overlap: float = 0.5,
) -> Tuple[int, ...]:
    """Return one occupancy label index per seat region."""
    pred = pred.filter(score_thresh)
    labels = []
    for region in seat_regions:
        frac = overlap_fraction(pred.boxes, region)
        mask = frac >= min_overlap
        if not mask.any():
            labels.append(EMPTY)
            continue
        rank = np.where(mask, pred.scores * frac, -1.0)
        best = int(np.argmax(rank))
        labels.append(detection_to_occupancy(int(pred.labels[best])))
    return tuple(labels)


def occupancy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    """``y_true``/``y_pred``: ``(num_frames, num_seats)`` occupancy indices."""
    n_cls = len(OCCUPANCY_CLASSES)
    confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true.ravel(), y_pred.ravel()):
        confusion[t, p] += 1
    metrics: Dict[str, object] = {
        "seat_accuracy": float((y_true == y_pred).mean()) if y_true.size else 0.0,
        "frame_accuracy": float((y_true == y_pred).all(axis=1).mean()) if y_true.size else 0.0,
    }
    for s, pos in enumerate(SEAT_POSITIONS):
        metrics[f"seat_accuracy/{pos}"] = float((y_true[:, s] == y_pred[:, s]).mean()) if y_true.size else 0.0
    for c, name in enumerate(OCCUPANCY_CLASSES):
        tp = confusion[c, c]
        support = confusion[c].sum()
        predicted = confusion[:, c].sum()
        metrics[f"recall/{name}"] = float(tp / support) if support else 0.0
        metrics[f"precision/{name}"] = float(tp / predicted) if predicted else 0.0

    child_true = np.isin(y_true, list(CHILD_CLASSES))
    child_pred = np.isin(y_pred, list(CHILD_CLASSES))
    tp = int((child_true & child_pred).sum())
    metrics["child_presence_recall"] = float(tp / child_true.sum()) if child_true.sum() else 0.0
    metrics["child_presence_precision"] = float(tp / child_pred.sum()) if child_pred.sum() else 0.0
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def evaluate_occupancy(
    ann_file: str | Path,
    predictions: Mapping[int, Prediction],
    layout: SeatLayout,
) -> Dict[str, object]:
    """Score per-seat occupancy for every image in ``predictions`` against the exported labels."""
    images = {im["id"]: im for im in json.loads(Path(ann_file).read_text())["images"]}
    y_true, y_pred = [], []
    for image_id, pred in predictions.items():
        info = images[image_id]
        if "seat_labels" not in info:
            continue
        regions = layout.seats_for(info.get("vehicle"), info["width"], info["height"])
        y_true.append(info["seat_labels"])
        y_pred.append(assign_seats(pred, regions, layout.score_thresh, layout.min_overlap))
    n = len(SEAT_POSITIONS)
    return occupancy_metrics(
        np.asarray(y_true, dtype=np.int64).reshape(-1, n), np.asarray(y_pred, dtype=np.int64).reshape(-1, n)
    )
