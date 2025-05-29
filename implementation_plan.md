# CabinSense — Implementation Plan

Vehicle seat occupancy detection benchmark on the **SVIRO** synthetic vehicle-interior dataset.
Five detector families (YOLOv8, YOLOX, Faster R-CNN, RetinaNet, EfficientDet) are trained and
evaluated under one harness, with MLflow experiment tracking and optional GPU training on
AWS SageMaker. Detections are mapped to rear-seat positions to produce a per-seat occupancy
verdict (empty / infant seat / child seat / person / everyday object), the signal used for
child-presence and airbag/seat-belt reminder applications.

Reference result from the original study: **55.70% mAP@50 with Faster R-CNN** (best of five).

## Key design decisions

- **Single canonical annotation format.** SVIRO ships per-image bounding-box text files.
  We parse them once into an internal `Sample` record and export to COCO JSON (torchvision,
  EfficientDet, YOLOX, pycocotools evaluation) and YOLO txt (Ultralytics).
- **Uniform `Detector` interface.** Every model family is wrapped in an adapter exposing
  `fit()`, `predict()` and `save()/load()` so the benchmark runner never special-cases models.
  Families with their own training engines (Ultralytics, YOLOX) delegate to them; torchvision
  and EfficientDet share an in-house PyTorch training loop.
- **One evaluator for all models.** All predictions are converted to COCO detections and scored
  with pycocotools so mAP@50 / mAP@50:95 are directly comparable.
- **Occupancy layer is model-agnostic.** Seat regions are defined per vehicle in YAML; detections
  are assigned to seats by IoU-over-seat, producing per-seat accuracy alongside mAP.
- **Config-driven.** YAML configs per model + a benchmark config listing which runs to execute.
- **Local or SageMaker.** The same `cabinsense train` entry point runs locally (CUDA if
  available) or inside a SageMaker PyTorch training job launched by `cabinsense sagemaker`.

## Phases

### Phase 1 — Project scaffold and SVIRO data pipeline
Deliverables: package layout, config loading, SVIRO parser, COCO/YOLO exporters, PyTorch
dataset + transforms, deterministic train/val split by vehicle.
Files: `pyproject.toml`, `requirements.txt`, `cabinsense/__init__.py`, `cabinsense/config.py`,
`cabinsense/data/{__init__,classes,sviro,converters,dataset,transforms}.py`,
`configs/data/sviro.yaml`, `scripts/prepare_sviro.py`.

### Phase 2 — Model zoo behind a common interface
Deliverables: `Detector` base class + registry; adapters for Faster R-CNN and RetinaNet
(torchvision), EfficientDet (effdet), YOLOv8 (ultralytics) and YOLOX (megvii yolox).
Files: `cabinsense/models/{__init__,base,registry,torchvision_models,efficientdet,yolov8,yolox_model}.py`,
`configs/models/*.yaml`.

### Phase 3 — Training loop, evaluation and occupancy metrics
Deliverables: shared PyTorch trainer (AMP, warmup + cosine LR, checkpointing, early stopping),
COCO evaluator (mAP@50, mAP@50:95, per-class AP), seat-occupancy assignment and per-seat
accuracy/confusion matrix.
Files: `cabinsense/training/{__init__,trainer,schedulers,utils}.py`,
`cabinsense/evaluation/{__init__,coco_eval,occupancy}.py`, `configs/seats/*.yaml`.

### Phase 4 — MLflow tracking, benchmark runner, SageMaker launcher and CLI
Deliverables: MLflow logger (params, per-epoch metrics, artifacts, model registry tag),
benchmark orchestrator that runs every configured model and emits a leaderboard
(Markdown + CSV), SageMaker estimator launcher + in-container entry script, `cabinsense` CLI.
Files: `cabinsense/tracking/mlflow_logger.py`, `cabinsense/benchmark/{runner,report}.py`,
`cabinsense/cloud/{sagemaker_launcher,entrypoint}.py`, `cabinsense/cli.py`,
`configs/benchmark.yaml`, `.env.example`.

### Phase 5 — Tests, inference demo and polish
Deliverables: pytest suite on synthetic SVIRO-like fixtures (parser, converters, occupancy
assignment, evaluator, registry), single-image inference + visualization script, Makefile.
Files: `tests/*`, `scripts/infer.py`, `cabinsense/visualize.py`, `Makefile`.

After phases: `.gitignore`, `README.md` with architecture diagram, run and test instructions.
