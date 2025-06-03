"""YOLOv8 via Ultralytics.

Ultralytics owns its training loop, augmentation and NMS, so this adapter only translates
configuration in and results out. Per-epoch metrics are read back from ``results.csv`` and
forwarded to the CabinSense tracker so every model shows up identically in MLflow.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import DataBundle, Detector, ImageInput, Prediction, load_image
from .registry import register

try:
    from ultralytics import YOLO
    from ultralytics import settings as ul_settings
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("YOLOv8 requires `pip install ultralytics` (or cabinsense[yolov8])") from exc

# Ultralytics keys -> CabinSense metric names.
_METRIC_MAP = {
    "metrics/mAP50(B)": "val/mAP50",
    "metrics/mAP50-95(B)": "val/mAP50_95",
    "metrics/precision(B)": "val/precision",
    "metrics/recall(B)": "val/recall",
    "train/box_loss": "train/box_loss",
    "train/cls_loss": "train/cls_loss",
    "train/dfl_loss": "train/dfl_loss",
}


@register("yolov8")
class YOLOv8Detector(Detector):
    def __init__(self, cfg: Dict[str, Any], num_classes: int, device: Optional[str] = None):
        super().__init__(cfg, num_classes, device)
        # We log to MLflow ourselves; avoid duplicate runs from the Ultralytics callback.
        ul_settings.update({"mlflow": False})
        self.model = YOLO(cfg.get("weights", "yolov8m.pt"))
        self.image_size = int(cfg.get("image_size", 640))

    def _device_arg(self) -> str:
        return "0" if self.device.startswith("cuda") else "cpu"

    def fit(self, data: DataBundle, output_dir: Path, tracker: Any = None) -> Dict[str, float]:
        tcfg = dict(self.cfg.get("train", {}))
        output_dir = Path(output_dir)
        self.model.train(
            data=str(data.yolo_yaml),
            epochs=int(tcfg.pop("epochs", 50)),
            imgsz=self.image_size,
            batch=int(tcfg.pop("batch_size", 16)),
            device=self._device_arg(),
            project=str(output_dir),
            name="ultralytics",
            exist_ok=True,
            seed=int(tcfg.pop("seed", 0)),
            patience=int(tcfg.pop("patience", 10)),
            workers=int(tcfg.pop("num_workers", 4)),
            **tcfg,  # remaining keys (lr0, optimizer, mosaic, ...) pass through verbatim
        )
        run_dir = output_dir / "ultralytics"
        history = self._read_results(run_dir / "results.csv")
        if tracker is not None:
            for epoch, row in enumerate(history):
                tracker.log_metrics(row, step=epoch)
        best = run_dir / "weights" / "best.pt"
        if best.exists():
            shutil.copy2(best, output_dir / "best.pt")
            self.model = YOLO(str(best))
        if not history:
            return {}
        best_row = max(history, key=lambda r: r.get("val/mAP50_95", 0.0))
        return {k: v for k, v in best_row.items() if k.startswith("val/")}

    @staticmethod
    def _read_results(path: Path) -> List[Dict[str, float]]:
        if not path.exists():
            return []
        rows = []
        with open(path, newline="") as fh:
            for raw in csv.DictReader(fh):
                row = {}
                for key, value in raw.items():
                    key = key.strip()
                    if key in _METRIC_MAP and value not in (None, ""):
                        row[_METRIC_MAP[key]] = float(value)
                rows.append(row)
        return rows

    def predict(self, images: List[ImageInput], score_thresh: float = 0.001) -> List[Prediction]:
        pil_images = [load_image(im) for im in images]
        results = self.model.predict(
            pil_images,
            imgsz=self.image_size,
            conf=score_thresh,
            device=self._device_arg(),
            verbose=False,
            max_det=int(self.cfg.get("max_detections", 100)),
        )
        preds = []
        for res in results:
            boxes = res.boxes
            preds.append(
                Prediction(
                    boxes=boxes.xyxy.cpu().numpy().astype(np.float32),
                    scores=boxes.conf.cpu().numpy().astype(np.float32),
                    labels=boxes.cls.cpu().numpy().astype(np.int64),
                )
            )
        return preds

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        return path

    def load(self, path: Path) -> None:
        self.model = YOLO(str(path))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.model.parameters())
