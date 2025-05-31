"""Parser for the SVIRO synthetic vehicle-interior dataset.

Expected layout (as distributed by DFKI)::

    <root>/<vehicle>/<split>/<image_type>/<vehicle>_<split>_imageID_<n>_GT_<l>_<m>_<r>[_suffix].png
    <root>/<vehicle>/<split>/<box_dir>/<vehicle>_<split>_imageID_<n>_GT_<l>_<m>_<r>[_suffix].npy

Box files contain one row per object: ``[class_id, x1, y1, x2, y2]`` (``.npy``) or the same
values whitespace/comma separated (``.txt``). Images and boxes are matched on ``imageID``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .classes import SVIRO_TO_DETECTION, SVIRO_TO_OCCUPANCY

log = logging.getLogger(__name__)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
BOX_EXTS = {".npy", ".txt", ".csv"}

_NAME_RE = re.compile(r"imageID_(?P<id>\d+)_GT_(?P<l>\d)_(?P<m>\d)_(?P<r>\d)")


@dataclass
class Box:
    """Axis-aligned box in absolute pixel coordinates, detection class index."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: int

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def as_xyxy(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Sample:
    """One SVIRO frame with detection boxes and per-seat occupancy labels."""

    image_path: Path
    vehicle: str
    split: str
    image_id: int
    seat_labels: Tuple[int, int, int]  # occupancy indices for left/middle/right
    boxes: List[Box] = field(default_factory=list)
    width: int = 0
    height: int = 0

    @property
    def key(self) -> str:
        return f"{self.vehicle}/{self.split}/{self.image_id}"


def parse_filename(name: str) -> Optional[Tuple[int, Tuple[int, int, int]]]:
    """Return ``(image_id, raw_seat_codes)`` or ``None`` if the name is not a SVIRO frame."""
    match = _NAME_RE.search(name)
    if not match:
        return None
    seats = (int(match["l"]), int(match["m"]), int(match["r"]))
    return int(match["id"]), seats


def load_box_file(path: Path) -> np.ndarray:
    """Load a SVIRO box file into an ``(N, 5)`` float array ``[cls, x1, y1, x2, y2]``."""
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        text = path.read_text().replace(",", " ").strip()
        if not text:
            return np.zeros((0, 5), dtype=np.float32)
        arr = np.array([[float(v) for v in line.split()] for line in text.splitlines() if line.strip()])
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[1] < 5:
        raise ValueError(f"{path}: expected 5 columns [cls,x1,y1,x2,y2], got {arr.shape[1]}")
    return arr[:, :5]


def convert_boxes(raw: np.ndarray, min_size: float = 0.0) -> List[Box]:
    boxes: List[Box] = []
    for cls_id, x1, y1, x2, y2 in raw:
        label = SVIRO_TO_DETECTION.get(int(cls_id))
        if label is None:
            continue
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        boxes.append(Box(x1, y1, x2, y2, label))
    return boxes


def _index_by_id(directory: Path, exts: Iterable[str]) -> Dict[int, Path]:
    index: Dict[int, Path] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in exts:
            continue
        parsed = parse_filename(path.name)
        if parsed is not None:
            index[parsed[0]] = path
    return index


def _image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(path) as img:
        return img.size


class SVIRODataset:
    """Indexes SVIRO frames across vehicles and splits."""

    def __init__(
        self,
        root: str | Path,
        image_type: str = "grayscale_wholeImage",
        box_dir: str = "boxes_wholeImage",
        min_box_size: float = 0.0,
        read_sizes: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_type = image_type
        self.box_dir = box_dir
        self.min_box_size = min_box_size
        self.read_sizes = read_sizes

    def available_vehicles(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.is_dir() else []

    def load_vehicle(self, vehicle: str, split: str) -> List[Sample]:
        split_dir = self.root / vehicle / split
        images = _index_by_id(split_dir / self.image_type, IMAGE_EXTS)
        boxes = _index_by_id(split_dir / self.box_dir, BOX_EXTS)
        samples: List[Sample] = []
        missing = 0
        for image_id, image_path in images.items():
            _, raw_seats = parse_filename(image_path.name)  # type: ignore[misc]
            seat_labels = tuple(SVIRO_TO_OCCUPANCY.get(c, 0) for c in raw_seats)
            box_path = boxes.get(image_id)
            if box_path is None:
                missing += 1
                det_boxes: List[Box] = []
            else:
                det_boxes = convert_boxes(load_box_file(box_path), self.min_box_size)
            width, height = _image_size(image_path) if self.read_sizes else (0, 0)
            samples.append(
                Sample(
                    image_path=image_path,
                    vehicle=vehicle,
                    split=split,
                    image_id=image_id,
                    seat_labels=seat_labels,  # type: ignore[arg-type]
                    boxes=det_boxes,
                    width=width,
                    height=height,
                )
            )
        if missing:
            log.warning("%s/%s: %d images without a box file", vehicle, split, missing)
        return samples

    def load(self, vehicles: Sequence[str], split: str) -> List[Sample]:
        out: List[Sample] = []
        for vehicle in vehicles:
            out.extend(self.load_vehicle(vehicle, split))
        log.info("Loaded %d SVIRO samples (%s, %d vehicles)", len(out), split, len(vehicles))
        return out


def build_splits(cfg: dict) -> Dict[str, List[Sample]]:
    """Create train/val/test sample lists from a data config.

    Validation holds out entire vehicles (``val_vehicles``) from SVIRO's train split; the test
    split uses SVIRO's own test folders for every configured vehicle.
    """
    ds = SVIRODataset(
        cfg["root"],
        image_type=cfg.get("image_type", "grayscale_wholeImage"),
        box_dir=cfg.get("box_dir", "boxes_wholeImage"),
        min_box_size=float(cfg.get("min_box_size", 0)),
    )
    vehicles = cfg.get("vehicles") or ds.available_vehicles()
    val_vehicles = set(cfg.get("val_vehicles", []))
    train_vehicles = [v for v in vehicles if v not in val_vehicles]
    return {
        "train": ds.load(train_vehicles, "train"),
        "val": ds.load(sorted(val_vehicles), "train"),
        "test": ds.load(vehicles, cfg.get("test_split", "test")),
    }
