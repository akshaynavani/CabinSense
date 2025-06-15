#!/usr/bin/env python
"""Run a trained CabinSense model on one or more cabin images and print the seat verdict.

    python scripts/infer.py --model configs/models/faster_rcnn.yaml \
        --checkpoint outputs/faster_rcnn/final.pt --vehicle aclass path/to/frame.png [--out viz/]
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from cabinsense.config import load_config
from cabinsense.data.classes import NUM_CLASSES, OCCUPANCY_CLASSES, SEAT_POSITIONS
from cabinsense.evaluation.occupancy import SeatLayout, assign_seats
from cabinsense.models import build_detector
from cabinsense.visualize import draw, save


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seats", default="configs/seats/default.yaml")
    ap.add_argument("--vehicle", default=None, help="vehicle key for seat layout overrides")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="directory for annotated images")
    args = ap.parse_args()

    detector = build_detector(load_config(args.model), num_classes=NUM_CLASSES, device=args.device)
    detector.load(Path(args.checkpoint))
    layout = SeatLayout.from_yaml(args.seats)

    preds = detector.predict(args.images, score_thresh=0.05)
    for path, pred in zip(args.images, preds):
        img = Image.open(path)
        regions = layout.seats_for(args.vehicle, img.width, img.height)
        seats = assign_seats(pred, regions, layout.score_thresh, layout.min_overlap)
        verdict = {pos: OCCUPANCY_CLASSES[s] for pos, s in zip(SEAT_POSITIONS, seats)}
        child = any(OCCUPANCY_CLASSES[s] in ("infant_seat", "child_seat") for s in seats)
        print(json.dumps({"image": path, "seats": verdict, "child_present": child}))
        if args.out:
            out = Path(args.out) / (Path(path).stem + "_cabinsense.png")
            save(draw(img, pred, regions, seats, layout.score_thresh), out)


if __name__ == "__main__":
    main()
