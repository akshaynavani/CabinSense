"""EfficientDet via Ross Wightman's ``effdet`` package.

effdet expects a fixed square input, boxes in ``yxyx`` order and 1-based class ids. Training uses
``DetBenchTrain`` (anchor labelling + focal loss); inference wraps the same backbone/BiFPN in
``DetBenchPredict``.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from ..data.transforms import Compose, build_transforms
from .base import Prediction
from .registry import register
from .torch_base import TorchDetector

try:
    from effdet import DetBenchPredict, create_model
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("EfficientDet requires `pip install effdet timm` (or cabinsense[efficientdet])") from exc


@register("efficientdet")
class EfficientDetDetector(TorchDetector):
    @property
    def image_size(self) -> int:
        return int(self.cfg.get("image_size", 512))

    def build_model(self) -> nn.Module:
        bench = create_model(
            self.cfg.get("arch", "tf_efficientdet_d0"),
            bench_task="train",
            num_classes=self.num_classes,
            pretrained=bool(self.cfg.get("pretrained", True)),
            image_size=(self.image_size, self.image_size),
            max_det_per_image=int(self.cfg.get("max_detections", 100)),
        )
        self._predict_bench = None
        return bench

    @property
    def predict_bench(self) -> nn.Module:
        if self._predict_bench is None:
            self._predict_bench = DetBenchPredict(self.model.model).to(self.device)
        return self._predict_bench

    def train_transforms(self) -> Compose:
        return build_transforms(train=True, image_size=self.image_size, normalize=True)

    def eval_transforms(self) -> Compose:
        return build_transforms(train=False, image_size=self.image_size, normalize=True)

    def compute_loss(self, images, targets) -> Dict[str, torch.Tensor]:
        x = torch.stack(images).to(self.device, non_blocking=True)
        boxes, classes = [], []
        for t in targets:
            b = t["boxes"].to(self.device)
            boxes.append(b[:, [1, 0, 3, 2]] if b.numel() else b.reshape(0, 4))
            classes.append((t["labels"] + 1).float().to(self.device))
        out = self.model(x, {"bbox": boxes, "cls": classes})
        return {"class_loss": out["class_loss"], "box_loss": out["box_loss"]}

    @torch.no_grad()
    def predict_batch(self, images, targets) -> List[Prediction]:
        x = torch.stack(images).to(self.device, non_blocking=True)
        bench = self.predict_bench
        bench.eval()
        dets = bench(x)  # (B, max_det, 6): x1, y1, x2, y2, score, cls(1-based)
        preds = []
        for det, tgt in zip(dets, targets):
            det = det[det[:, 4] > 0]
            scale = float(tgt.get("scale", torch.tensor(1.0)))
            preds.append(
                Prediction(
                    boxes=(det[:, :4] / scale).float().cpu().numpy(),
                    scores=det[:, 4].float().cpu().numpy(),
                    labels=(det[:, 5].long() - 1).cpu().numpy(),
                )
            )
        return preds
