import json

import numpy as np
import pytest

from cabinsense.data.converters import write_coco
from cabinsense.data.sviro import Box, Sample
from cabinsense.evaluation.coco_eval import evaluate_coco, predictions_to_coco
from cabinsense.evaluation.occupancy import (
    SeatLayout,
    assign_seats,
    evaluate_occupancy,
    occupancy_metrics,
    overlap_fraction,
)
from cabinsense.models.base import Prediction


@pytest.fixture()
def gt_file(tmp_path):
    samples = [
        Sample(tmp_path / "a.png", "aclass", "test", 1, (3, 0, 1), [Box(5, 10, 35, 90, 2), Box(70, 10, 98, 90, 0)], 100, 100),
        Sample(tmp_path / "b.png", "aclass", "test", 2, (0, 4, 0), [Box(40, 40, 60, 60, 3)], 100, 100),
    ]
    return write_coco(samples, tmp_path / "gt.json")


def _perfect(gt_file):
    data = json.loads(gt_file.read_text())
    preds = {im["id"]: [] for im in data["images"]}
    for a in data["annotations"]:
        x, y, w, h = a["bbox"]
        preds[a["image_id"]].append(([x, y, x + w, y + h], 0.9, a["category_id"] - 1))
    return {
        k: Prediction(
            np.array([b for b, _, _ in v], np.float32).reshape(-1, 4),
            np.array([s for _, s, _ in v], np.float32),
            np.array([c for _, _, c in v], np.int64),
        )
        for k, v in preds.items()
    }


def test_predictions_to_coco_format():
    pred = Prediction(np.array([[1, 2, 11, 22]], np.float32), np.array([0.5], np.float32), np.array([2]))
    (r,) = predictions_to_coco({7: pred})
    assert r == {"image_id": 7, "category_id": 3, "bbox": [1.0, 2.0, 10.0, 20.0], "score": 0.5}


def test_perfect_predictions_score_one(gt_file):
    m = evaluate_coco(gt_file, _perfect(gt_file))
    assert m["mAP50"] == pytest.approx(1.0)
    assert m["mAP50_95"] == pytest.approx(1.0)
    assert m["AP50/person"] == pytest.approx(1.0)


def test_no_predictions_score_zero(gt_file):
    m = evaluate_coco(gt_file, {1: Prediction.empty(), 2: Prediction.empty()})
    assert m["mAP50"] == 0.0


def test_overlap_fraction():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    frac = overlap_fraction(boxes, np.array([0, 0, 10, 10], np.float32))
    assert frac.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_assign_seats_picks_best_candidate_and_defaults_empty():
    regions = [np.array([0, 0, 40, 100]), np.array([33, 0, 67, 100]), np.array([60, 0, 100, 100])]
    pred = Prediction(
        np.array([[5, 10, 35, 90], [6, 10, 34, 90], [70, 10, 98, 90]], np.float32),
        np.array([0.6, 0.9, 0.2], np.float32),
        np.array([3, 2, 0], np.int64),
    )
    # left: person (0.9) beats object (0.6); middle: nothing; right: below score threshold
    assert assign_seats(pred, regions, score_thresh=0.4, min_overlap=0.5) == (3, 0, 0)
    assert assign_seats(pred, regions, score_thresh=0.1, min_overlap=0.5) == (3, 0, 1)


def test_occupancy_metrics_child_presence():
    y_true = np.array([[1, 0, 3], [2, 0, 0]])
    y_pred = np.array([[1, 0, 3], [0, 0, 2]])
    m = occupancy_metrics(y_true, y_pred)
    assert m["seat_accuracy"] == pytest.approx(4 / 6)
    assert m["frame_accuracy"] == pytest.approx(0.5)
    assert m["child_presence_recall"] == pytest.approx(0.5)
    assert m["child_presence_precision"] == pytest.approx(0.5)
    assert np.asarray(m["confusion_matrix"]).sum() == 6


def test_evaluate_occupancy_end_to_end(gt_file, tmp_path):
    layout_file = tmp_path / "seats.yaml"
    layout_file.write_text(
        "default:\n  left: [0, 0, 0.4, 1]\n  middle: [0.33, 0, 0.67, 1]\n  right: [0.6, 0, 1, 1]\n"
        "assignment:\n  score_thresh: 0.3\n  min_overlap: 0.5\n"
    )
    m = evaluate_occupancy(gt_file, _perfect(gt_file), SeatLayout.from_yaml(layout_file))
    assert m["seat_accuracy"] == pytest.approx(1.0)
    assert m["frame_accuracy"] == pytest.approx(1.0)
