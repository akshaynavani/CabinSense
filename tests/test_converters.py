import json

import pytest

from cabinsense.data.converters import export_split, to_coco, write_yolo_dataset_yaml, yolo_lines
from cabinsense.data.dataset import CocoDetectionDataset
from cabinsense.data.prepare import prepare
from cabinsense.data.sviro import Box, Sample, build_splits
from cabinsense.data.transforms import RandomHorizontalFlip, build_transforms


def _sample(tmp_path):
    return Sample(tmp_path / "x.png", "aclass", "train", 1, (3, 0, 0), [Box(10, 20, 30, 60, 2)], 100, 80)


def test_to_coco_uses_one_based_categories_and_xywh(tmp_path):
    coco = to_coco([_sample(tmp_path)])
    (ann,) = coco["annotations"]
    assert ann["category_id"] == 3
    assert ann["bbox"] == [10, 20, 20, 40]
    assert ann["area"] == 800
    assert [c["id"] for c in coco["categories"]] == [1, 2, 3, 4]
    assert coco["images"][0]["seat_labels"] == [3, 0, 0]


def test_yolo_lines_are_normalised(tmp_path):
    (line,) = yolo_lines(_sample(tmp_path))
    cls, cx, cy, w, h = line.split()
    assert cls == "2"
    assert float(cx) == pytest.approx(0.2)
    assert float(cy) == pytest.approx(0.5)
    assert float(w) == pytest.approx(0.2)
    assert float(h) == pytest.approx(0.5)


def test_yolo_lines_require_image_size(tmp_path):
    s = _sample(tmp_path)
    s.width = 0
    with pytest.raises(ValueError):
        yolo_lines(s)


def test_export_split_layout(data_cfg, tmp_path):
    samples = build_splits(data_cfg)["train"]
    export = tmp_path / "export"
    paths = export_split(samples, export, "train", link_mode="copy")
    assert len(list(paths["images"].iterdir())) == 3
    assert len(list(paths["labels"].iterdir())) == 3
    data = json.loads(paths["annotations"].read_text())
    assert all("/" not in im["file_name"] for im in data["images"])
    yaml_path = write_yolo_dataset_yaml(export, ["train"])
    assert "infant_seat" in yaml_path.read_text()


def test_prepare_and_dataset_roundtrip(data_cfg):
    export = prepare(data_cfg, link_mode="copy")
    stats = json.loads((export / "stats.json").read_text())
    assert stats["train"]["images"] == 3
    assert stats["val"]["boxes"] == 2

    ds = CocoDetectionDataset(export / "images" / "train", export / "annotations" / "train.json", build_transforms(False))
    image, target = ds[0]
    assert image.shape[0] == 3  # grayscale promoted to RGB
    assert target["boxes"].shape[1] == 4
    assert target["labels"].min() >= 0
    assert target["seat_labels"].tolist() == [3, 0, 1]


def test_horizontal_flip_mirrors_boxes_and_swaps_seats():
    import torch
    from PIL import Image

    img = Image.new("RGB", (100, 50))
    target = {"boxes": torch.tensor([[10.0, 0, 30, 20]]), "seat_labels": torch.tensor([3, 0, 1])}
    _, out = RandomHorizontalFlip(p=1.0)(img, target)
    assert out["boxes"].tolist() == [[70.0, 0, 90, 20]]
    assert out["seat_labels"].tolist() == [1, 0, 3]


def test_resize_records_scale():
    import torch
    from PIL import Image

    tf = build_transforms(train=False, image_size=50)
    img = Image.new("L", (100, 40))
    t, target = tf(img, {"boxes": torch.tensor([[10.0, 10, 20, 20]]), "labels": torch.tensor([0])})
    assert tuple(t.shape) == (3, 50, 50)
    assert float(target["scale"]) == pytest.approx(0.5)
    assert target["boxes"].tolist() == [[5.0, 5.0, 10.0, 10.0]]
