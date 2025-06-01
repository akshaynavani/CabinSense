"""Shared plumbing for detectors trained with CabinSense's own PyTorch loop."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..data.dataset import CocoDetectionDataset, make_loader
from ..data.transforms import Compose
from .base import DataBundle, Detector, ImageInput, Prediction, load_image


class TorchDetector(Detector):
    """Detector whose model is a plain ``nn.Module`` trained by :class:`cabinsense.training.Trainer`."""

    def __init__(self, cfg: Dict[str, Any], num_classes: int, device: Optional[str] = None):
        super().__init__(cfg, num_classes, device)
        self.model: nn.Module = self.build_model().to(self.device)

    # ---- hooks for subclasses ---------------------------------------------------------------
    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        ...

    @abc.abstractmethod
    def train_transforms(self) -> Compose:
        ...

    @abc.abstractmethod
    def eval_transforms(self) -> Compose:
        ...

    @abc.abstractmethod
    def compute_loss(self, images: List[torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        """Return a dict of scalar losses; the trainer back-propagates their sum."""

    @abc.abstractmethod
    def predict_batch(self, images: List[torch.Tensor], targets: List[Dict]) -> List[Prediction]:
        """Inference on transformed images; must return boxes in original image coordinates."""

    # ---- shared behaviour -------------------------------------------------------------------
    def parameter_groups(self) -> List[Dict[str, Any]]:
        """Backbone gets a reduced LR multiplier (``train.backbone_lr_mult``)."""
        mult = float(self.cfg.get("train", {}).get("backbone_lr_mult", 1.0))
        backbone, rest = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone if "backbone" in name else rest).append(p)
        groups = [{"params": rest, "lr_mult": 1.0}]
        if backbone:
            groups.append({"params": backbone, "lr_mult": mult})
        return groups

    def build_loaders(self, data: DataBundle):
        tcfg = self.cfg.get("train", {})
        bs = int(tcfg.get("batch_size", 8))
        workers = int(tcfg.get("num_workers", 4))
        train_ds = CocoDetectionDataset(data.images("train"), data.annotations("train"), self.train_transforms())
        val_ds = CocoDetectionDataset(data.images("val"), data.annotations("val"), self.eval_transforms())
        return (
            make_loader(train_ds, bs, shuffle=True, num_workers=workers),
            make_loader(val_ds, int(tcfg.get("eval_batch_size", bs)), shuffle=False, num_workers=workers),
        )

    def fit(self, data: DataBundle, output_dir: Path, tracker: Any = None) -> Dict[str, float]:
        from ..training.trainer import Trainer

        train_loader, val_loader = self.build_loaders(data)
        trainer = Trainer(
            detector=self,
            cfg=self.cfg.get("train", {}),
            output_dir=Path(output_dir),
            val_ann_file=data.annotations("val"),
            tracker=tracker,
        )
        return trainer.fit(train_loader, val_loader)

    @torch.no_grad()
    def predict(self, images: List[ImageInput], score_thresh: float = 0.001) -> List[Prediction]:
        self.model.eval()
        tf = self.eval_transforms()
        tensors, targets = [], []
        for img in images:
            pil = load_image(img)
            target = {
                "boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.tensor([pil.height, pil.width]),
                "scale": torch.tensor(1.0),
            }
            t, target = tf(pil, target)
            tensors.append(t)
            targets.append(target)
        preds = self.predict_batch(tensors, targets)
        return [p.filter(score_thresh) for p in preds]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"family": self.family, "cfg": self.cfg, "state_dict": self.model.state_dict()}, path)
        return path

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
