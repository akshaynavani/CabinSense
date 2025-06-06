"""Faster R-CNN and RetinaNet from torchvision with heads resized for SVIRO's classes.

torchvision detectors reserve label 0 for background, so dataset labels are shifted by +1 on
the way in and -1 on the way out.
"""

from __future__ import annotations

from functools import partial
from typing import Dict, List

import torch
from torch import nn
from torchvision.models import detection as tvd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from ..data.transforms import Compose, build_transforms
from .base import Prediction
from .registry import register
from .torch_base import TorchDetector

_FASTER_RCNN = {
    "fasterrcnn_resnet50_fpn": tvd.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": tvd.fasterrcnn_resnet50_fpn_v2,
    "fasterrcnn_mobilenet_v3_large_fpn": tvd.fasterrcnn_mobilenet_v3_large_fpn,
}
_RETINANET = {
    "retinanet_resnet50_fpn": tvd.retinanet_resnet50_fpn,
    "retinanet_resnet50_fpn_v2": tvd.retinanet_resnet50_fpn_v2,
}


def _shift_targets(targets: List[Dict], device: str) -> List[Dict]:
    out = []
    for t in targets:
        out.append({"boxes": t["boxes"].to(device), "labels": (t["labels"] + 1).to(device)})
    return out


@register("torchvision")
class TorchvisionDetector(TorchDetector):
    """``model.arch`` selects any Faster R-CNN or RetinaNet variant listed above."""

    def build_model(self) -> nn.Module:
        arch = self.cfg["arch"]
        pretrained = bool(self.cfg.get("pretrained", True))
        weights = "DEFAULT" if pretrained else None
        kwargs = {
            "min_size": int(self.cfg.get("min_size", 640)),
            "max_size": int(self.cfg.get("max_size", 1024)),
        }
        n_out = self.num_classes + 1  # + background
        if arch in _FASTER_RCNN:
            model = _FASTER_RCNN[arch](weights=weights, **kwargs)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_out)
            model.roi_heads.detections_per_img = int(self.cfg.get("max_detections", 100))
        elif arch in _RETINANET:
            model = _RETINANET[arch](weights=weights, **kwargs)
            head = model.head.classification_head
            norm = partial(nn.GroupNorm, 32) if arch.endswith("_v2") else None
            new_head = RetinaNetClassificationHead(
                model.backbone.out_channels, head.num_anchors, n_out, norm_layer=norm
            )
            model.head.classification_head = new_head
            model.detections_per_img = int(self.cfg.get("max_detections", 100))
            model.score_thresh = float(self.cfg.get("score_thresh", 0.05))
        else:
            raise ValueError(f"Unsupported torchvision arch {arch!r}")
        if self.cfg.get("freeze_backbone_bn", True):
            for m in model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad_(False)
        return model

    def train_mode(self) -> None:
        # model.train() would re-enable running-stat updates in the frozen BatchNorms; with
        # batch size 8 those statistics are too noisy, so keep them in eval mode.
        self.model.train()
        if self.cfg.get("freeze_backbone_bn", True):
            for m in self.model.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def train_transforms(self) -> Compose:
        # Resizing and normalisation happen inside GeneralizedRCNNTransform.
        return build_transforms(train=True)

    def eval_transforms(self) -> Compose:
        return build_transforms(train=False)

    def compute_loss(self, images, targets) -> Dict[str, torch.Tensor]:
        images = [im.to(self.device, non_blocking=True) for im in images]
        return self.model(images, _shift_targets(targets, self.device))

    @torch.no_grad()
    def predict_batch(self, images, targets) -> List[Prediction]:
        images = [im.to(self.device, non_blocking=True) for im in images]
        outputs = self.model(images)
        preds = []
        for out, tgt in zip(outputs, targets):
            scale = float(tgt.get("scale", torch.tensor(1.0)))
            preds.append(
                Prediction(
                    boxes=(out["boxes"] / scale).float().cpu().numpy(),
                    scores=out["scores"].float().cpu().numpy(),
                    labels=(out["labels"] - 1).cpu().numpy().astype("int64"),
                )
            )
        return preds
