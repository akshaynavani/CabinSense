"""Class definitions shared by detection and seat-occupancy evaluation.

SVIRO encodes per-seat ground truth in the file name (``..._GT_<left>_<middle>_<right>``)
with the following codes:

    0 empty seat, 1 infant seat (with baby), 2 child seat (with child), 3 adult person,
    4 everyday object, 5 empty infant seat, 6 empty child seat

Bounding boxes use the same object ids. For detection we collapse the "empty seat"
variants into their seat category so a detector learns four foreground classes.
"""

from __future__ import annotations

from typing import Dict, List

# Detection classes (contiguous ids starting at 0, background excluded).
DETECTION_CLASSES: List[str] = ["infant_seat", "child_seat", "person", "everyday_object"]
NUM_CLASSES = len(DETECTION_CLASSES)

# Raw SVIRO object id -> detection class index (None = ignore).
SVIRO_TO_DETECTION: Dict[int, int | None] = {
    0: None,  # background / empty seat never has a box
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 0,
    6: 1,
}

# Seat occupancy labels (what the end application consumes).
OCCUPANCY_CLASSES: List[str] = [
    "empty",
    "infant_seat",
    "child_seat",
    "person",
    "everyday_object",
]

# Raw SVIRO seat code -> occupancy label index.
SVIRO_TO_OCCUPANCY: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2}

SEAT_POSITIONS: List[str] = ["left", "middle", "right"]


def detection_to_occupancy(det_class: int) -> int:
    """Map a detection class index to the occupancy label index."""
    return det_class + 1


def class_name(idx: int) -> str:
    return DETECTION_CLASSES[idx]
