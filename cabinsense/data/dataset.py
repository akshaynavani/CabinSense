"""PyTorch dataset reading the exported COCO annotations."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CocoDetectionDataset(Dataset):
    """Returns ``(image_tensor, target)`` with 0-based labels and xyxy boxes.

    The COCO ``image_id`` is preserved in the target so predictions can be scored with
    pycocotools against the same annotation file.
    """

    def __init__(self, image_dir: str | Path, ann_file: str | Path, transforms: Optional[Callable] = None):
        self.image_dir = Path(image_dir)
        self.ann_file = Path(ann_file)
        self.transforms = transforms
        data = json.loads(self.ann_file.read_text())
        self.images: List[Dict] = sorted(data["images"], key=lambda im: im["id"])
        self.anns: Dict[int, List[Dict]] = defaultdict(list)
        for ann in data["annotations"]:
            self.anns[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def image_path(self, idx: int) -> Path:
        name = self.images[idx]["file_name"]
        path = Path(name)
        return path if path.is_absolute() else self.image_dir / name

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        info = self.images[idx]
        image = Image.open(self.image_path(idx))
        anns = self.anns.get(info["id"], [])
        boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor([a["category_id"] - 1 for a in anns], dtype=torch.int64),
            "image_id": torch.tensor(info["id"], dtype=torch.int64),
            "orig_size": torch.tensor([info["height"], info["width"]], dtype=torch.int64),
            "scale": torch.tensor(1.0),
        }
        if "seat_labels" in info:
            target["seat_labels"] = torch.tensor(info["seat_labels"], dtype=torch.int64)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
    )
