"""SageMaker training-job entry point (copied to ``train.py`` in the job's source dir).

SageMaker conventions used:
    SM_CHANNEL_SVIRO  mounted exported dataset (annotations/, images/, labels/, data.yaml)
    SM_MODEL_DIR      contents are tarred to model.tar.gz in S3 when the job ends
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cabinsense.benchmark.runner import run_model  # noqa: E402
from cabinsense.config import load_config  # noqa: E402
from cabinsense.data.converters import write_yolo_dataset_yaml  # noqa: E402
from cabinsense.evaluation.occupancy import SeatLayout  # noqa: E402
from cabinsense.models import DataBundle  # noqa: E402
from cabinsense.tracking import build_tracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True)
    p.add_argument("--bench-config", default="configs/benchmark.yaml")
    p.add_argument("--overrides", default="", help="space separated key=value overrides for the model config")
    p.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_SVIRO", "data/sviro_export"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "outputs"))
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    root = Path(__file__).resolve().parent
    bench = load_config(root / args.bench_config)
    model_cfg = load_config(root / args.model_config, args.overrides.split() if args.overrides else [])
    bench["output_dir"] = args.model_dir

    data_dir = Path(args.data_dir)
    # data.yaml written locally holds an absolute path from the workstation; rewrite it for the container.
    write_yolo_dataset_yaml(data_dir, [s for s in ("train", "val", "test") if (data_dir / "images" / s).exists()])
    data = DataBundle(data_dir)
    layout = SeatLayout.from_yaml(root / bench["seats"]) if bench.get("seats") else None
    tracker = build_tracker(bench)
    summary = run_model(model_cfg, bench, data, tracker=tracker, layout=layout, train=True)
    logging.info("Finished %s: %s", summary["model"], {k: v for k, v in summary.items() if "mAP" in k})


if __name__ == "__main__":
    main()
