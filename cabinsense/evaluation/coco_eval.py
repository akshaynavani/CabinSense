"""COCO-style mAP for any detector, via pycocotools.

Every family's predictions are converted into the same COCO results format and scored against
the same exported annotation file, which is what makes the leaderboard comparable.
"""

from __future__ import annotations

import contextlib
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..data.classes import DETECTION_CLASSES
from ..models.base import Detector, Prediction


def predictions_to_coco(predictions: Mapping[int, Prediction]) -> List[Dict]:
    """``{coco_image_id: Prediction}`` -> COCO result dicts (1-based category ids, xywh)."""
    results = []
    for image_id, pred in predictions.items():
        for (x1, y1, x2, y2), score, label in zip(pred.boxes, pred.scores, pred.labels):
            results.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label) + 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    return results


def _load_gt(ann: str | Path | COCO) -> COCO:
    if isinstance(ann, COCO):
        return ann
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ann))


def evaluate_coco(
    ann: str | Path | COCO,
    predictions: Mapping[int, Prediction],
    class_names: Sequence[str] = DETECTION_CLASSES,
    verbose: bool = False,
) -> Dict[str, float]:
    """Return mAP@50, mAP@50:95, mAP@75, AR@100 and per-class AP@50 / AP@50:95."""
    gt = _load_gt(ann)
    results = predictions_to_coco(predictions)
    metrics: Dict[str, float] = {"mAP50": 0.0, "mAP50_95": 0.0, "mAP75": 0.0, "AR100": 0.0}
    for name in class_names:
        metrics[f"AP50/{name}"] = 0.0
        metrics[f"AP50_95/{name}"] = 0.0
    if not results:
        return metrics

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        dt = gt.loadRes(results)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.params.imgIds = sorted(predictions.keys())
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    if verbose:
        print(sink.getvalue())

    stats = ev.stats
    metrics.update(
        {"mAP50_95": float(stats[0]), "mAP50": float(stats[1]), "mAP75": float(stats[2]), "AR100": float(stats[8])}
    )
    # precision: [T iou thresholds, R recall pts, K classes, A areas, M max dets]
    precision = ev.eval["precision"]
    cat_ids = gt.getCatIds()
    for k, cat_id in enumerate(cat_ids):
        name = gt.loadCats(cat_id)[0]["name"]
        p_all = precision[:, :, k, 0, -1]
        p_50 = precision[0, :, k, 0, -1]
        metrics[f"AP50_95/{name}"] = float(np.mean(p_all[p_all > -1])) if (p_all > -1).any() else 0.0
        metrics[f"AP50/{name}"] = float(np.mean(p_50[p_50 > -1])) if (p_50 > -1).any() else 0.0
    return metrics


def predict_split(
    detector: Detector,
    image_dir: str | Path,
    ann_file: str | Path,
    batch_size: int = 8,
    score_thresh: float = 0.001,
    limit: Optional[int] = None,
) -> Dict[int, Prediction]:
    """Run ``detector.predict`` over every image in an exported split (family-agnostic)."""
    images = json.loads(Path(ann_file).read_text())["images"]
    if limit:
        images = images[:limit]
    out: Dict[int, Prediction] = {}
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        paths = [Path(image_dir) / im["file_name"] for im in chunk]
        for im, pred in zip(chunk, detector.predict(paths, score_thresh=score_thresh)):
            out[im["id"]] = pred
    return out


def benchmark_latency(detector: Detector, image_paths: Sequence[Path], warmup: int = 3, runs: int = 20) -> Dict[str, float]:
    """Single-image latency (ms) and throughput, including pre/post-processing."""
    if not image_paths:
        return {}
    paths = list(image_paths)
    for i in range(warmup):
        detector.predict([paths[i % len(paths)]])
    timings = []
    for i in range(runs):
        _sync()
        t0 = time.perf_counter()
        detector.predict([paths[i % len(paths)]])
        _sync()
        timings.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(timings)
    return {"latency_ms_mean": float(arr.mean()), "latency_ms_p95": float(np.percentile(arr, 95)), "fps": float(1000.0 / arr.mean())}


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass
