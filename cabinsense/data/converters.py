"""Export parsed SVIRO samples to COCO JSON and Ultralytics YOLO formats."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import yaml

from .classes import DETECTION_CLASSES
from .sviro import Sample


def sample_file_name(sample: Sample) -> str:
    return f"{sample.vehicle}_{sample.split}_{sample.image_id:05d}{sample.image_path.suffix}"


def to_coco(samples: Sequence[Sample], image_prefix: str = "") -> Dict:
    """Build a COCO detection dict. Category ids are 1-based (COCO convention)."""
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    for img_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": img_id,
                "file_name": os.path.join(image_prefix, sample_file_name(sample)) if image_prefix
                else str(sample.image_path),
                "width": sample.width,
                "height": sample.height,
                "vehicle": sample.vehicle,
                "seat_labels": list(sample.seat_labels),
            }
        )
        for box in sample.boxes:
            w, h = box.width, box.height
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": box.label + 1,
                    "bbox": [box.x1, box.y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    categories = [
        {"id": i + 1, "name": name, "supercategory": "cabin"} for i, name in enumerate(DETECTION_CLASSES)
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def write_coco(samples: Sequence[Sample], out_path: Path, image_prefix: str = "") -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(to_coco(samples, image_prefix), fh)
    return out_path


def yolo_lines(sample: Sample) -> List[str]:
    """Normalised ``cls cx cy w h`` lines for a sample."""
    if not sample.width or not sample.height:
        raise ValueError(f"{sample.key}: image size required for YOLO export")
    lines = []
    for box in sample.boxes:
        cx = (box.x1 + box.x2) / 2.0 / sample.width
        cy = (box.y1 + box.y2) / 2.0 / sample.height
        w = box.width / sample.width
        h = box.height / sample.height
        lines.append(f"{box.label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _place_image(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def export_split(
    samples: Sequence[Sample], export_dir: Path, split: str, link_mode: str = "symlink"
) -> Dict[str, Path]:
    """Materialise one split in a layout both COCO- and YOLO-based trainers understand::

        export_dir/images/<split>/*.png
        export_dir/labels/<split>/*.txt          (YOLO)
        export_dir/annotations/<split>.json      (COCO, file_name relative to images/<split>)
    """
    img_dir = export_dir / "images" / split
    lbl_dir = export_dir / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        name = sample_file_name(sample)
        _place_image(sample.image_path, img_dir / name, link_mode)
        (lbl_dir / (Path(name).stem + ".txt")).write_text("\n".join(yolo_lines(sample)))
    ann_path = write_coco(samples, export_dir / "annotations" / f"{split}.json", image_prefix=".")
    # COCO file names are relative to img_dir; normalise "./x.png" -> "x.png".
    data = json.loads(ann_path.read_text())
    for img in data["images"]:
        img["file_name"] = os.path.basename(img["file_name"])
    ann_path.write_text(json.dumps(data))
    return {"images": img_dir, "labels": lbl_dir, "annotations": ann_path}


def write_yolo_dataset_yaml(export_dir: Path, splits: Sequence[str] = ("train", "val", "test")) -> Path:
    """Write the ``data.yaml`` that Ultralytics expects."""
    payload = {"path": str(export_dir.resolve()), "names": {i: n for i, n in enumerate(DETECTION_CLASSES)}}
    for split in splits:
        payload[split] = f"images/{split}"
    out = export_dir / "data.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out
