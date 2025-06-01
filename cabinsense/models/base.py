"""Common interface every benchmarked detector implements."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

ImageInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class Prediction:
    """Detections for one image in original pixel coordinates (xyxy, 0-based class ids)."""

    boxes: np.ndarray  # (N, 4) float32
    scores: np.ndarray  # (N,) float32
    labels: np.ndarray  # (N,) int64

    @classmethod
    def empty(cls) -> "Prediction":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64))

    def filter(self, score_thresh: float) -> "Prediction":
        keep = self.scores >= score_thresh
        return Prediction(self.boxes[keep], self.scores[keep], self.labels[keep])

    def __len__(self) -> int:
        return int(self.scores.shape[0])


@dataclass
class DataBundle:
    """Paths to an exported SVIRO dataset (see ``cabinsense.data.converters.export_split``)."""

    root: Path
    splits: Sequence[str] = ("train", "val", "test")
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def images(self, split: str) -> Path:
        return self.root / "images" / split

    def annotations(self, split: str) -> Path:
        return self.root / "annotations" / f"{split}.json"

    @property
    def yolo_yaml(self) -> Path:
        return self.root / "data.yaml"


def load_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.open(image).convert("RGB")


class Detector(abc.ABC):
    """Adapter around one detector family.

    ``fit`` trains on ``data.train`` (validating on ``data.val``) and returns a dict of final
    metrics; ``predict`` runs inference on raw images and returns :class:`Prediction` objects in
    original image coordinates so a single evaluator can score every model.
    """

    family: str = "base"

    def __init__(self, cfg: Dict[str, Any], num_classes: int, device: Optional[str] = None):
        self.cfg = cfg
        self.num_classes = num_classes
        self.device = device or self._default_device()
        self.name = cfg.get("name", self.family)

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:  # pragma: no cover
            return "cpu"

    @abc.abstractmethod
    def fit(self, data: DataBundle, output_dir: Path, tracker: Any = None) -> Dict[str, float]:
        ...

    @abc.abstractmethod
    def predict(self, images: List[ImageInput], score_thresh: float = 0.001) -> List[Prediction]:
        ...

    @abc.abstractmethod
    def save(self, path: Path) -> Path:
        ...

    @abc.abstractmethod
    def load(self, path: Path) -> None:
        ...

    def num_parameters(self) -> int:
        return 0
