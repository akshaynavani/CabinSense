"""Turn a raw SVIRO download into the exported train/val/test layout."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict

from .classes import DETECTION_CLASSES, OCCUPANCY_CLASSES
from .converters import export_split, write_yolo_dataset_yaml
from .sviro import build_splits

log = logging.getLogger(__name__)


def summarise(samples) -> Dict:
    box_counts = Counter(DETECTION_CLASSES[b.label] for s in samples for b in s.boxes)
    seat_counts = Counter(OCCUPANCY_CLASSES[l] for s in samples for l in s.seat_labels)
    return {
        "images": len(samples),
        "boxes": sum(box_counts.values()),
        "boxes_per_class": dict(box_counts),
        "seat_occupancy": dict(seat_counts),
        "vehicles": sorted({s.vehicle for s in samples}),
    }


def prepare(cfg: dict, link_mode: str = "symlink") -> Path:
    export_dir = Path(cfg["export_dir"])
    splits = build_splits(cfg)
    stats = {}
    for name, samples in splits.items():
        if not samples:
            log.warning("Split %s is empty; check root=%s", name, cfg["root"])
            continue
        export_split(samples, export_dir, name, link_mode=link_mode)
        stats[name] = summarise(samples)
        log.info("Exported %s: %d images, %d boxes", name, stats[name]["images"], stats[name]["boxes"])
    write_yolo_dataset_yaml(export_dir, [s for s in splits if splits[s]])
    (export_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return export_dir
