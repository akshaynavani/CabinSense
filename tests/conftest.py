"""Synthetic SVIRO-like fixtures: tiny images, real file naming, real box format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

WIDTH, HEIGHT = 96, 64

# (vehicle, split, image_id, seat codes, [[cls, x1, y1, x2, y2], ...])
FRAMES = [
    ("aclass", "train", 1, (3, 0, 1), [[3, 2, 10, 30, 60], [1, 70, 12, 94, 58]]),
    ("aclass", "train", 2, (0, 4, 0), [[4, 40, 30, 56, 50]]),
    ("aclass", "train", 3, (0, 0, 0), []),
    ("x5", "train", 1, (2, 0, 6), [[2, 4, 8, 34, 62], [6, 66, 10, 92, 60]]),
    ("aclass", "test", 7, (5, 3, 0), [[5, 3, 9, 33, 61], [3, 38, 8, 58, 63]]),
]


def _name(vehicle, split, image_id, seats, ext):
    l, m, r = seats
    return f"{vehicle}_{split}_imageID_{image_id}_GT_{l}_{m}_{r}{ext}"


@pytest.fixture()
def sviro_root(tmp_path: Path) -> Path:
    root = tmp_path / "sviro"
    for vehicle, split, image_id, seats, boxes in FRAMES:
        img_dir = root / vehicle / split / "grayscale_wholeImage"
        box_dir = root / vehicle / split / "boxes_wholeImage"
        img_dir.mkdir(parents=True, exist_ok=True)
        box_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((HEIGHT, WIDTH), 128, dtype=np.uint8)).save(
            img_dir / _name(vehicle, split, image_id, seats, ".png")
        )
        np.save(box_dir / _name(vehicle, split, image_id, seats, ".npy"), np.asarray(boxes, dtype=np.float32).reshape(-1, 5))
    return root


@pytest.fixture()
def data_cfg(sviro_root: Path, tmp_path: Path) -> dict:
    return {
        "root": str(sviro_root),
        "image_type": "grayscale_wholeImage",
        "box_dir": "boxes_wholeImage",
        "vehicles": ["aclass", "x5"],
        "val_vehicles": ["x5"],
        "test_split": "test",
        "min_box_size": 2,
        "export_dir": str(tmp_path / "export"),
    }
