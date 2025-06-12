"""``cabinsense`` command-line interface.

    cabinsense prepare    --data configs/data/sviro.yaml [--copy]
    cabinsense train      --model configs/models/faster_rcnn.yaml
    cabinsense evaluate   --model configs/models/faster_rcnn.yaml --checkpoint outputs/faster_rcnn/final.pt
    cabinsense benchmark  [--only faster_rcnn yolov8m]
    cabinsense report     --output-dir outputs
    cabinsense sagemaker  upload --export-dir data/sviro_export
    cabinsense sagemaker  launch --data-uri s3://bucket/cabinsense/sviro_export [--only yolov8m]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import load_config


def _bench(args) -> dict:
    return load_config(args.bench, getattr(args, "set", None) or [])


def cmd_prepare(args) -> None:
    from .data.prepare import prepare

    out = prepare(load_config(args.data, args.set), link_mode="copy" if args.copy else "symlink")
    print(json.dumps(json.loads((out / "stats.json").read_text()), indent=2))


def _single(args, train: bool) -> None:
    from .benchmark.runner import run_model
    from .evaluation.occupancy import SeatLayout
    from .models import DataBundle
    from .tracking import NullTracker, build_tracker

    bench = _bench(args)
    data_cfg = load_config(bench["data"])
    model_cfg = load_config(args.model, args.model_set or [])
    tracker = NullTracker() if args.no_mlflow else build_tracker(bench)
    layout = SeatLayout.from_yaml(bench["seats"]) if bench.get("seats") else None
    summary = run_model(
        model_cfg,
        bench,
        DataBundle(data_cfg["export_dir"]),
        tracker=tracker,
        layout=layout,
        train=train,
        checkpoint=Path(args.checkpoint) if getattr(args, "checkpoint", None) else None,
    )
    print(json.dumps(summary, indent=2, default=str))


def cmd_train(args) -> None:
    _single(args, train=True)


def cmd_evaluate(args) -> None:
    _single(args, train=False)


def cmd_benchmark(args) -> None:
    from .benchmark.runner import run_benchmark

    results = run_benchmark(_bench(args), only=args.only)
    out = Path(_bench(args).get("output_dir", "outputs")) / "leaderboard.md"
    if out.exists():
        print(out.read_text())
    if not results:
        sys.exit(1)


def cmd_report(args) -> None:
    from .benchmark.report import load_summaries, write_leaderboard

    summaries = load_summaries(Path(args.output_dir))
    if not summaries:
        sys.exit(f"No */summary.json found under {args.output_dir}")
    paths = write_leaderboard(summaries, Path(args.output_dir), split=args.split)
    print(paths["markdown"].read_text())


def cmd_sagemaker(args) -> None:
    from .cloud import sagemaker_launcher as sm

    if args.action == "upload":
        print(sm.upload_dataset(Path(args.export_dir), prefix=args.prefix))
    else:
        if not args.data_uri:
            sys.exit("--data-uri is required for launch")
        for job in sm.launch_benchmark(args.bench, args.data_uri, only=args.only, wait=args.wait):
            print(job)


def cmd_families(_args) -> None:
    from .models import available_families

    print("\n".join(available_families()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cabinsense", description="SVIRO seat occupancy detection benchmark")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prepare", help="export raw SVIRO to COCO + YOLO layouts")
    sp.add_argument("--data", default="configs/data/sviro.yaml")
    sp.add_argument("--copy", action="store_true", help="copy images (needed before uploading to S3)")
    sp.add_argument("--set", nargs="*", default=[], help="data config overrides key=value")
    sp.set_defaults(func=cmd_prepare)

    for name, func, helptext in (
        ("train", cmd_train, "train + evaluate a single model"),
        ("evaluate", cmd_evaluate, "evaluate a trained checkpoint"),
    ):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--model", required=True, help="model config yaml")
        sp.add_argument("--bench", default="configs/benchmark.yaml")
        sp.add_argument("--set", nargs="*", default=[], help="benchmark config overrides")
        sp.add_argument("--model-set", nargs="*", default=[], help="model config overrides, e.g. train.epochs=1")
        sp.add_argument("--no-mlflow", action="store_true")
        if name == "evaluate":
            sp.add_argument("--checkpoint", required=True)
        sp.set_defaults(func=func)

    sp = sub.add_parser("benchmark", help="train + evaluate every configured model")
    sp.add_argument("--bench", default="configs/benchmark.yaml")
    sp.add_argument("--only", nargs="*", help="restrict to these model names or families")
    sp.add_argument("--set", nargs="*", default=[])
    sp.set_defaults(func=cmd_benchmark)

    sp = sub.add_parser("report", help="rebuild leaderboard from saved summaries")
    sp.add_argument("--output-dir", default="outputs")
    sp.add_argument("--split", default="test")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("sagemaker", help="upload data / launch SageMaker training jobs")
    sp.add_argument("action", choices=["upload", "launch"])
    sp.add_argument("--bench", default="configs/benchmark.yaml")
    sp.add_argument("--export-dir", default="data/sviro_export")
    sp.add_argument("--prefix", default="cabinsense/sviro_export")
    sp.add_argument("--data-uri")
    sp.add_argument("--only", nargs="*")
    sp.add_argument("--wait", action="store_true")
    sp.set_defaults(func=cmd_sagemaker)

    sp = sub.add_parser("families", help="list registered detector families")
    sp.set_defaults(func=cmd_families)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
