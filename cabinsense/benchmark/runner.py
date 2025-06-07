"""Train and evaluate every configured detector, one MLflow run per model."""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import load_config
from ..data.classes import NUM_CLASSES
from ..evaluation.coco_eval import benchmark_latency, evaluate_coco, predict_split
from ..evaluation.occupancy import SeatLayout, evaluate_occupancy
from ..models import DataBundle, build_detector
from ..models.base import Detector
from ..tracking import NullTracker, build_tracker
from .report import write_leaderboard

log = logging.getLogger(__name__)


def _environment_tags() -> Dict[str, str]:
    tags = {"python": platform.python_version(), "host": platform.node()}
    try:
        import torch

        tags["torch"] = torch.__version__
        if torch.cuda.is_available():
            tags["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    return tags


def evaluate_detector(
    detector: Detector,
    data: DataBundle,
    split: str,
    eval_cfg: Dict[str, Any],
    layout: Optional[SeatLayout],
    out_dir: Path,
) -> Dict[str, Any]:
    """Run inference on ``split`` and compute COCO, occupancy and latency metrics."""
    ann = data.annotations(split)
    preds = predict_split(
        detector,
        data.images(split),
        ann,
        batch_size=int(eval_cfg.get("batch_size", 8)),
        score_thresh=float(eval_cfg.get("score_thresh", 0.001)),
        limit=eval_cfg.get("limit"),
    )
    metrics: Dict[str, Any] = dict(evaluate_coco(ann, preds))
    if layout is not None:
        occ = evaluate_occupancy(ann, preds, layout)
        (out_dir / f"{split}_confusion.json").write_text(json.dumps(occ.pop("confusion_matrix")))
        metrics.update({f"occupancy/{k}": v for k, v in occ.items()})
    runs = int(eval_cfg.get("latency_runs", 0))
    if runs:
        sample_paths = [data.images(split) / im["file_name"] for im in json.loads(ann.read_text())["images"][:runs]]
        metrics.update(benchmark_latency(detector, sample_paths, runs=runs))
    dump = {
        str(k): {"boxes": p.boxes.tolist(), "scores": p.scores.tolist(), "labels": p.labels.tolist()}
        for k, p in preds.items()
    }
    (out_dir / f"{split}_predictions.json").write_text(json.dumps(dump))
    return metrics


def run_model(
    model_cfg: Dict[str, Any],
    bench_cfg: Dict[str, Any],
    data: DataBundle,
    tracker=None,
    layout: Optional[SeatLayout] = None,
    train: bool = True,
    checkpoint: Optional[Path] = None,
) -> Dict[str, Any]:
    tracker = tracker or NullTracker()
    name = model_cfg.get("name", model_cfg["family"])
    out_dir = Path(bench_cfg.get("output_dir", "outputs")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg = bench_cfg.get("eval", {})
    split = eval_cfg.get("split", "test")

    detector = build_detector(model_cfg, num_classes=NUM_CLASSES, device=bench_cfg.get("device"))
    tags = {"family": model_cfg["family"], "model": name, "dataset": "SVIRO", **_environment_tags()}
    with tracker.start_run(run_name=name, tags=tags):
        tracker.log_params({"model": model_cfg, "eval": eval_cfg, "data_root": str(data.root)})
        tracker.log_metrics({"params_millions": detector.num_parameters() / 1e6})
        t0 = time.time()
        if checkpoint:
            detector.load(Path(checkpoint))
            val_metrics: Dict[str, Any] = {}
        elif train:
            val_metrics = detector.fit(data, out_dir, tracker=tracker)
        else:
            raise ValueError("Pass train=True or a checkpoint to evaluate")
        train_time = time.time() - t0

        test_metrics = evaluate_detector(detector, data, split, eval_cfg, layout, out_dir)
        tracker.log_metrics({f"{split}/{k}": v for k, v in test_metrics.items()})
        tracker.log_metrics({"train_time_min": train_time / 60.0})

        ckpt = detector.save(out_dir / "final.pt")
        summary = {
            "model": name,
            "family": model_cfg["family"],
            "params_m": detector.num_parameters() / 1e6,
            "train_time_min": train_time / 60.0,
            **{k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
            **{f"{split}/{k}": v for k, v in test_metrics.items()},
            "mlflow_run_id": tracker.run_id,
            "checkpoint": str(ckpt),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        tracker.log_artifact(out_dir / "summary.json")
        tracker.log_artifact(out_dir / f"{split}_confusion.json", artifact_path="eval")
        if bench_cfg.get("tracking", {}).get("log_checkpoints", True):
            tracker.log_artifact(ckpt, artifact_path="checkpoints")
    log.info("%s: %s mAP50=%.4f mAP50:95=%.4f", name, split, test_metrics["mAP50"], test_metrics["mAP50_95"])
    return summary


def run_benchmark(bench_cfg: Dict[str, Any], only: Optional[List[str]] = None, train: bool = True) -> List[Dict[str, Any]]:
    """Run every model listed in ``bench_cfg['models']`` and write the leaderboard."""
    data_cfg = load_config(bench_cfg["data"]) if isinstance(bench_cfg["data"], str) else bench_cfg["data"]
    data = DataBundle(bench_cfg.get("export_dir") or data_cfg["export_dir"])
    layout = SeatLayout.from_yaml(bench_cfg["seats"]) if bench_cfg.get("seats") else None
    tracker = build_tracker(bench_cfg)

    results: List[Dict[str, Any]] = []
    for model_path in bench_cfg["models"]:
        model_cfg = load_config(model_path, bench_cfg.get("model_overrides", []))
        if only and model_cfg.get("name") not in only and model_cfg["family"] not in only:
            continue
        try:
            results.append(run_model(model_cfg, bench_cfg, data, tracker, layout, train=train))
        except Exception:  # keep benchmarking the remaining models
            log.exception("Model %s failed", model_cfg.get("name"))
            if bench_cfg.get("fail_fast", False):
                raise
    out_dir = Path(bench_cfg.get("output_dir", "outputs"))
    if results:
        paths = write_leaderboard(results, out_dir, split=bench_cfg.get("eval", {}).get("split", "test"))
        log.info("Leaderboard written to %s", paths["markdown"])
    return results
