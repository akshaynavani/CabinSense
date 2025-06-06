"""Shared training loop for :class:`~cabinsense.models.torch_base.TorchDetector` models.

Features: mixed precision, linear warmup + cosine decay, gradient clipping, periodic COCO
evaluation on the validation split, best/last checkpoints and early stopping on ``monitor``.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from tqdm import tqdm

from ..evaluation.coco_eval import evaluate_coco
from .schedulers import WarmupScheduler, build_optimizer
from .utils import AverageMeter, seed_everything

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        detector,
        cfg: Dict[str, Any],
        output_dir: Path,
        val_ann_file: Path,
        tracker: Any = None,
    ) -> None:
        self.detector = detector
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_ann_file = Path(val_ann_file)
        self.tracker = tracker
        self.epochs = int(cfg.get("epochs", 12))
        self.eval_every = int(cfg.get("eval_every", 1))
        self.patience = int(cfg.get("patience", 0))  # 0 disables early stopping
        self.monitor = cfg.get("monitor", "mAP50_95")
        self.grad_clip = float(cfg.get("grad_clip", 0.0))
        self.log_every = int(cfg.get("log_every", 50))
        self.use_amp = bool(cfg.get("amp", True)) and detector.device.startswith("cuda")
        self.max_iters_per_epoch: Optional[int] = cfg.get("max_iters_per_epoch")
        seed_everything(int(cfg.get("seed", 0)))

    # -----------------------------------------------------------------------------------------
    def fit(self, train_loader, val_loader) -> Dict[str, float]:
        det = self.detector
        optimizer = build_optimizer(det.parameter_groups(), self.cfg)
        iters_per_epoch = len(train_loader) if not self.max_iters_per_epoch else min(len(train_loader), int(self.max_iters_per_epoch))
        scheduler = WarmupScheduler(
            optimizer,
            total_iters=self.epochs * iters_per_epoch,
            warmup_iters=int(self.cfg.get("warmup_iters", 500)),
            mode=self.cfg.get("scheduler", "cosine"),
        )
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        best_score, best_metrics, stale = -math.inf, {}, 0
        global_step = 0
        for epoch in range(self.epochs):
            det.train_mode()
            meters: Dict[str, AverageMeter] = {}
            t0 = time.time()
            pbar = tqdm(train_loader, total=iters_per_epoch, desc=f"[{det.name}] epoch {epoch + 1}/{self.epochs}", leave=False)
            for it, (images, targets) in enumerate(pbar):
                if it >= iters_per_epoch:
                    break
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    losses = det.compute_loss(images, targets)
                    loss = sum(losses.values())
                if not torch.isfinite(loss):
                    log.warning("Non-finite loss at epoch %d iter %d; skipping batch", epoch, it)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    continue
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(det.model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1

                for k, v in {**losses, "loss": loss}.items():
                    meters.setdefault(k, AverageMeter()).update(float(v.detach()), len(images))
                if global_step % self.log_every == 0:
                    pbar.set_postfix(loss=f"{meters['loss'].avg:.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    self._log({"train/loss_step": float(loss.detach()), "train/lr": optimizer.param_groups[0]["lr"]}, global_step)

            epoch_metrics = {f"train/{k}": m.avg for k, m in meters.items()}
            epoch_metrics["train/epoch_time_s"] = time.time() - t0

            if (epoch + 1) % self.eval_every == 0 or epoch + 1 == self.epochs:
                val = self.evaluate(val_loader)
                epoch_metrics.update({f"val/{k}": v for k, v in val.items()})
                score = val.get(self.monitor, 0.0)
                log.info("[%s] epoch %d  loss=%.4f  val mAP50=%.4f  mAP50:95=%.4f", det.name, epoch + 1,
                         epoch_metrics.get("train/loss", float("nan")), val["mAP50"], val["mAP50_95"])
                if score > best_score:
                    best_score, best_metrics, stale = score, {**val, "best_epoch": epoch + 1}, 0
                    det.save(self.output_dir / "best.pt")
                else:
                    stale += 1
            self._log(epoch_metrics, epoch)
            det.save(self.output_dir / "last.pt")
            if self.patience and stale >= self.patience:
                log.info("Early stopping after %d epochs without %s improvement", stale, self.monitor)
                break

        if (self.output_dir / "best.pt").exists():
            det.load(self.output_dir / "best.pt")
        (self.output_dir / "best_val_metrics.json").write_text(json.dumps(best_metrics, indent=2))
        return {f"val/{k}": v for k, v in best_metrics.items()}

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.detector.model.eval()
        preds = {}
        for images, targets in tqdm(loader, desc="eval", leave=False):
            for tgt, pred in zip(targets, self.detector.predict_batch(images, targets)):
                preds[int(tgt["image_id"])] = pred
        return evaluate_coco(self.val_ann_file, preds)

    def _log(self, metrics: Dict[str, float], step: int) -> None:
        if self.tracker is not None:
            self.tracker.log_metrics(metrics, step=step)
