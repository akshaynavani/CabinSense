import numpy as np

from cabinsense.data.classes import OCCUPANCY_CLASSES
from cabinsense.data.sviro import SVIRODataset, build_splits, convert_boxes, load_box_file, parse_filename


def test_parse_filename_extracts_id_and_seats():
    assert parse_filename("tesla_train_imageID_42_GT_3_0_5_grayscale.png") == (42, (3, 0, 5))
    assert parse_filename("readme.txt") is None


def test_load_box_file_txt_and_npy(tmp_path):
    txt = tmp_path / "b.txt"
    txt.write_text("3 1 2 30 40\n1, 50, 5, 90, 60\n")
    arr = load_box_file(txt)
    assert arr.shape == (2, 5)
    npy = tmp_path / "b.npy"
    np.save(npy, arr)
    assert np.allclose(load_box_file(npy), arr)
    empty = tmp_path / "e.txt"
    empty.write_text("")
    assert load_box_file(empty).shape == (0, 5)


def test_convert_boxes_maps_classes_and_filters_small():
    raw = np.array([[3, 0, 0, 10, 10], [5, 0, 0, 10, 10], [0, 0, 0, 5, 5], [4, 0, 0, 1, 1]], dtype=np.float32)
    boxes = convert_boxes(raw, min_size=2)
    # person -> 2, empty infant seat -> infant_seat (0); background dropped; tiny object dropped
    assert [b.label for b in boxes] == [2, 0]


def test_convert_boxes_normalises_flipped_coordinates():
    (box,) = convert_boxes(np.array([[3, 10, 20, 2, 5]], dtype=np.float32))
    assert box.as_xyxy() == [2, 5, 10, 20]


def test_load_vehicle_reads_seat_labels_and_sizes(sviro_root):
    samples = SVIRODataset(sviro_root).load_vehicle("aclass", "train")
    assert len(samples) == 3
    first = next(s for s in samples if s.image_id == 1)
    assert [OCCUPANCY_CLASSES[i] for i in first.seat_labels] == ["person", "empty", "infant_seat"]
    assert (first.width, first.height) == (96, 64)
    assert len(first.boxes) == 2


def test_build_splits_holds_out_vehicles(data_cfg):
    splits = build_splits(data_cfg)
    assert {s.vehicle for s in splits["train"]} == {"aclass"}
    assert {s.vehicle for s in splits["val"]} == {"x5"}
    assert len(splits["test"]) == 1
