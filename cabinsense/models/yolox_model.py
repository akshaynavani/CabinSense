"""YOLOX via the Megvii reference implementation.

YOLOX is configured through ``Exp`` classes; :class:`SVIROExp` points the stock COCO pipeline at
the exported SVIRO layout (``annotations/<split>.json`` + ``images/<split>/``) and records each
evaluation so the results reach the CabinSense tracker. Training uses YOLOX's own ``Trainer``
(single GPU, CUDA required).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import DataBundle, Detector, ImageInput, Prediction, load_image
from .registry import register

try:
    from yolox.core import Trainer as YoloxTrainer
    from yolox.data import COCODataset, TrainTransform, ValTransform
    from yolox.exp import Exp as YoloxBaseExp
    from yolox.utils import postprocess
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "YOLOX requires `pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git`"
    ) from exc

# variant -> (depth, width) multipliers, matching the official exps.
_VARIANTS = {
    "yolox_s": (0.33, 0.50),
    "yolox_m": (0.67, 0.75),
    "yolox_l": (1.00, 1.00),
    "yolox_x": (1.33, 1.25),
}


class SVIROExp(YoloxBaseExp):
    def __init__(self, data_root: Path, num_classes: int, cfg: Dict[str, Any], output_dir: Path):
        super().__init__()
        variant = cfg.get("variant", "yolox_m")
        self.depth, self.width = _VARIANTS[variant]
        self.num_classes = num_classes
        self.data_dir = str(data_root)
        self.train_ann, self.val_ann, self.test_ann = "train.json", "val.json", "test.json"
        size = int(cfg.get("image_size", 640))
        self.input_size = (size, size)
        self.test_size = (size, size)
        tcfg = cfg.get("train", {})
        self.max_epoch = int(tcfg.get("epochs", 50))
        self.no_aug_epochs = int(tcfg.get("no_aug_epochs", 10))
        self.warmup_epochs = int(tcfg.get("warmup_epochs", 3))
        self.eval_interval = int(tcfg.get("eval_every", 1))
        self.basic_lr_per_img = float(tcfg.get("lr_per_img", 0.01 / 64.0))
        self.data_num_workers = int(tcfg.get("num_workers", 4))
        self.mosaic_prob = float(tcfg.get("mosaic_prob", 1.0))
        self.mixup_prob = float(tcfg.get("mixup_prob", 0.5))
        self.flip_prob = float(tcfg.get("flip_prob", 0.5))
        self.test_conf = float(cfg.get("score_thresh", 0.001))
        self.nmsthre = float(cfg.get("nms_thresh", 0.65))
        self.exp_name = "yolox"
        self.output_dir = str(output_dir)
        self.eval_history: List[Dict[str, float]] = []

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="images/train",
            img_size=self.input_size,
            preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        testdev = kwargs.get("testdev", False)
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name="images/test" if testdev else "images/val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=kwargs.get("legacy", False)),
        )

    def eval(self, *args, **kwargs):
        result = super().eval(*args, **kwargs)
        ap50_95, ap50 = float(result[0]), float(result[1])
        self.eval_history.append({"val/mAP50_95": ap50_95, "val/mAP50": ap50})
        return result


def _trainer_args(cfg: Dict[str, Any]) -> argparse.Namespace:
    tcfg = cfg.get("train", {})
    return argparse.Namespace(
        experiment_name="yolox",
        name=None,
        batch_size=int(tcfg.get("batch_size", 16)),
        devices=1,
        fp16=bool(tcfg.get("amp", True)),
        occupy=False,
        resume=False,
        ckpt=cfg.get("weights"),
        start_epoch=None,
        cache=None,
        logger="tensorboard",
        dist_backend="nccl",
        dist_url=None,
        num_machines=1,
        machine_rank=0,
    )


@register("yolox")
class YOLOXDetector(Detector):
    def __init__(self, cfg: Dict[str, Any], num_classes: int, device: Optional[str] = None):
        super().__init__(cfg, num_classes, device)
        self.exp = SVIROExp(Path("."), num_classes, cfg, Path("outputs"))
        self.model = self.exp.get_model().to(self.device)
        self.preproc = ValTransform(legacy=False)

    def fit(self, data: DataBundle, output_dir: Path, tracker: Any = None) -> Dict[str, float]:
        if not torch.cuda.is_available():
            raise RuntimeError("YOLOX training requires a CUDA device")
        output_dir = Path(output_dir)
        self.exp = SVIROExp(data.root, self.num_classes, self.cfg, output_dir)
        torch.cuda.set_device(0)
        trainer = YoloxTrainer(self.exp, _trainer_args(self.cfg))
        trainer.train()
        if tracker is not None:
            for epoch, row in enumerate(self.exp.eval_history):
                tracker.log_metrics(row, step=epoch)
        best = output_dir / "yolox" / "best_ckpt.pth"
        if best.exists():
            self.load(best)
        if not self.exp.eval_history:
            return {}
        return max(self.exp.eval_history, key=lambda r: r["val/mAP50_95"])

    @torch.no_grad()
    def predict(self, images: List[ImageInput], score_thresh: float = 0.001) -> List[Prediction]:
        self.model.eval()
        batch, ratios = [], []
        for im in images:
            bgr = np.ascontiguousarray(np.asarray(load_image(im))[:, :, ::-1])
            h, w = bgr.shape[:2]
            ratios.append(min(self.exp.test_size[0] / h, self.exp.test_size[1] / w))
            tensor, _ = self.preproc(bgr, None, self.exp.test_size)
            batch.append(torch.from_numpy(tensor))
        x = torch.stack(batch).float().to(self.device)
        outputs = postprocess(self.model(x), self.num_classes, score_thresh, self.exp.nmsthre)
        preds = []
        for out, ratio in zip(outputs, ratios):
            if out is None:
                preds.append(Prediction.empty())
                continue
            out = out.cpu().numpy()
            preds.append(
                Prediction(
                    boxes=(out[:, :4] / ratio).astype(np.float32),
                    scores=(out[:, 4] * out[:, 5]).astype(np.float32),
                    labels=out[:, 6].astype(np.int64),
                )
            )
        return preds

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg}, path)
        return path

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model = self.exp.get_model().to(self.device)
        self.model.load_state_dict(ckpt["model"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
