import pytest

from cabinsense.benchmark.report import leaderboard_frame, write_leaderboard
from cabinsense.models import Detector, available_families, get_detector_class, register
from cabinsense.tracking import NullTracker, flatten
from cabinsense.tracking.mlflow_logger import sanitize_key


def test_registry_lists_all_benchmarked_families():
    assert {"torchvision", "efficientdet", "yolov8", "yolox"} <= set(available_families())


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        get_detector_class("not_a_detector")


def test_register_custom_detector():
    @register("dummy_for_test")
    class Dummy(Detector):
        def fit(self, data, output_dir, tracker=None):
            return {}

        def predict(self, images, score_thresh=0.001):
            return []

        def save(self, path):
            return path

        def load(self, path):
            pass

    det = get_detector_class("dummy_for_test")({"name": "d"}, num_classes=4, device="cpu")
    assert det.family == "dummy_for_test" and det.name == "d"


def test_warmup_then_cosine_schedule():
    torch = pytest.importorskip("torch")
    from cabinsense.training.schedulers import WarmupScheduler, build_optimizer

    params = [torch.nn.Parameter(torch.zeros(1))]
    opt = build_optimizer([{"params": params}], {"lr": 0.1, "optimizer": "sgd"})
    sched = WarmupScheduler(opt, total_iters=100, warmup_iters=10, warmup_factor=0.0, min_lr_ratio=0.0)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    assert lrs[0] == pytest.approx(0.0)
    assert lrs[10] == pytest.approx(0.1)
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:11]))
    assert lrs[-1] < 0.001


def test_backbone_lr_multiplier_applied():
    torch = pytest.importorskip("torch")
    from cabinsense.training.schedulers import build_optimizer

    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr_mult": 1.0},
              {"params": [torch.nn.Parameter(torch.zeros(1))], "lr_mult": 0.1}]
    opt = build_optimizer(groups, {"lr": 0.01, "optimizer": "adamw"})
    assert [g["lr"] for g in opt.param_groups] == pytest.approx([0.01, 0.001])


def test_flatten_and_sanitize():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}
    assert sanitize_key("metrics/mAP50(B)") == "metrics/mAP50_B_"


def test_null_tracker_is_a_context_manager():
    t = NullTracker()
    with t.start_run("x") as run:
        run.log_metrics({"a": 1.0})
    assert t.run_id is None


def test_leaderboard_sorted_by_map50(tmp_path):
    results = [
        {"model": "retinanet", "family": "torchvision", "test/mAP50": 0.48, "test/mAP50_95": 0.30},
        {"model": "faster_rcnn", "family": "torchvision", "test/mAP50": 0.557, "test/mAP50_95": 0.34},
        {"model": "yolov8m", "family": "yolov8", "test/mAP50": 0.52, "test/mAP50_95": 0.36},
    ]
    df = leaderboard_frame(results)
    assert df["Model"].tolist() == ["faster_rcnn", "yolov8m", "retinanet"]
    paths = write_leaderboard(results, tmp_path)
    md = paths["markdown"].read_text()
    assert "55.70" in md
    assert paths["csv"].exists()
