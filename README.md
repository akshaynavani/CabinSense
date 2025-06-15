# CabinSense

**Vehicle seat occupancy detection benchmark on the SVIRO synthetic cabin dataset.**

CabinSense trains and evaluates five object-detector families — **Faster R-CNN, RetinaNet,
EfficientDet, YOLOv8 and YOLOX** — under one harness on
[SVIRO](https://sviro.kl.dfki.de) (Synthetic Vehicle Interior Rear seat Occupancy), then turns
their detections into a per-seat verdict (`empty`, `infant_seat`, `child_seat`, `person`,
`everyday_object`) for each of the three rear seats. Every run is tracked in **MLflow**, and the
same entry point can run locally or as a GPU training job on **AWS SageMaker**.

Use cases: child-left-behind alerts, seat-belt reminders, airbag suppression, and occupant
awareness for autonomous vehicles.

Reference result from the original study: **Faster R-CNN reached 55.70% mAP@50**, the best of
the five families.

## Features

- SVIRO parser that reads per-image box files and the seat labels encoded in file names
  (`..._GT_<left>_<middle>_<right>`), with a validation split that **holds out whole vehicles**
  so the score reflects cabins the model has never seen.
- One export (`images/`, `labels/`, `annotations/*.json`, `data.yaml`) that the COCO-based
  trainers (torchvision, effdet, YOLOX) and Ultralytics can all read.
- A `Detector` interface (`fit` / `predict` / `save` / `load`) with a registry, so the benchmark
  runner treats every family the same way.
- A shared PyTorch trainer for torchvision and EfficientDet: AMP, warmup + cosine LR, a lower LR
  for the backbone, gradient clipping, early stopping, and best/last checkpoints.
- One evaluator for every model: pycocotools mAP@50, mAP@50:95, mAP@75, AR@100 and per-class AP.
- Seat occupancy layer: per-seat and per-frame accuracy, a confusion matrix, and
  child-presence recall/precision.
- Leaderboard written as Markdown and CSV, with per-model MLflow runs that hold the parameters,
  per-epoch curves, the confusion matrix and checkpoints.
- SageMaker launcher that uploads only the code and configs and starts one GPU job per model.
  Spot instances are supported.

## Architecture

```mermaid
flowchart LR
    subgraph Data["Data pipeline (cabinsense.data)"]
        RAW[(SVIRO raw<br/>images + box files)] --> PARSE[sviro.py<br/>parse frames + GT seat codes]
        PARSE --> SPLIT[build_splits<br/>vehicle hold-out val]
        SPLIT --> EXPORT[converters.py<br/>COCO JSON + YOLO txt]
    end

    EXPORT --> BUNDLE[[DataBundle<br/>images/ labels/ annotations/ data.yaml]]

    subgraph Models["Model zoo (cabinsense.models)"]
        REG{registry}
        REG --> TV[torchvision<br/>Faster R-CNN / RetinaNet]
        REG --> ED[effdet<br/>EfficientDet]
        REG --> Y8[Ultralytics<br/>YOLOv8]
        REG --> YX[Megvii<br/>YOLOX]
        TV & ED --> TR[Shared Trainer<br/>AMP · warmup+cosine · early stop]
        Y8 --> UL[Ultralytics engine]
        YX --> YT[YOLOX Trainer]
    end

    BUNDLE --> REG
    TR & UL & YT --> PRED[Prediction<br/>xyxy · score · class]

    subgraph Eval["Evaluation (cabinsense.evaluation)"]
        PRED --> COCO[coco_eval<br/>mAP@50, mAP@50:95, per-class AP]
        PRED --> OCC[occupancy<br/>seat regions → per-seat verdict]
        PRED --> LAT[latency / FPS]
    end

    COCO & OCC & LAT --> RUN[benchmark.runner]
    RUN --> MLF[(MLflow<br/>params · curves · artifacts)]
    RUN --> LB[leaderboard.md / .csv]

    subgraph Cloud["AWS SageMaker (optional)"]
        LAUNCH[sagemaker_launcher<br/>one GPU job per model] --> ENTRY[entrypoint.py<br/>run_model in container]
    end
    BUNDLE -. S3 channel .-> ENTRY
    ENTRY --> MLF
```

### Repository layout

```
cabinsense/
  config.py                 YAML loader (${ENV:-default}, _base_ inheritance, key=value overrides)
  data/                     SVIRO parser, class maps, COCO/YOLO export, dataset, transforms
  models/                   Detector base, registry, torchvision / effdet / YOLOv8 / YOLOX adapters
  training/                 shared trainer, LR schedules, seeding
  evaluation/               COCO mAP, seat occupancy metrics, latency
  tracking/                 MLflow wrapper (+ NullTracker)
  benchmark/                multi-model runner and leaderboard report
  cloud/                    SageMaker launcher and in-container entrypoint
  cli.py                    `cabinsense` command
configs/
  data/sviro.yaml           dataset root, vehicles, hold-out vehicles
  models/*.yaml             one file per benchmarked model
  seats/default.yaml        normalised rear-seat regions (+ per-vehicle overrides)
  benchmark.yaml            which models to run, eval + tracking + SageMaker settings
scripts/                    prepare_sviro.py, infer.py
tests/                      pytest suite on synthetic SVIRO-like fixtures
```

## How to run

### 1. Install

Python 3.9+ is required. Training needs a CUDA GPU; YOLOX training **requires** CUDA.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"                       # torch, torchvision, ultralytics, effdet, mlflow, sagemaker
pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git   # YOLOX (only if benchmarking it)
cp .env.example .env                               # then edit paths / AWS settings
```

The model families are optional extras (`cabinsense[yolov8]`, `[efficientdet]`, `[yolox]`,
`[sagemaker]`). Each adapter is imported only when its family is used.

### 2. Get and export SVIRO

Download the *bounding box* and *whole image* archives for the vehicles you want from
<https://sviro.kl.dfki.de>, and extract them so the layout looks like this:

```
data/sviro/<vehicle>/<train|test>/grayscale_wholeImage/*.png
data/sviro/<vehicle>/<train|test>/boxes_wholeImage/*.npy   (or .txt)
```

Then export:

```bash
cabinsense prepare --data configs/data/sviro.yaml           # symlinks images
cabinsense prepare --data configs/data/sviro.yaml --copy    # copies them (needed for S3 upload)
```

This writes `data/sviro_export/` with `images/`, `labels/`, `annotations/{train,val,test}.json`,
`data.yaml` and `stats.json`. Edit `vehicles` / `val_vehicles` in `configs/data/sviro.yaml` to
change which cabins are held out.

### 3. Train and evaluate

```bash
# Quick check: one short epoch of Faster R-CNN on a slice of the data
make smoke

# One model
cabinsense train --model configs/models/faster_rcnn.yaml
cabinsense train --model configs/models/yolov8.yaml --model-set train.epochs=100

# Evaluate a saved checkpoint
cabinsense evaluate --model configs/models/faster_rcnn.yaml --checkpoint outputs/faster_rcnn/final.pt

# All five models, then the leaderboard
cabinsense benchmark                       # or: --only faster_rcnn yolov8m
cat outputs/leaderboard.md
```

Each model writes `outputs/<name>/` (`best.pt`, `final.pt`, `summary.json`,
`test_predictions.json`, `test_confusion.json`). `cabinsense report` rebuilds the leaderboard
from those summaries, for example after downloading SageMaker outputs.

### 4. MLflow

By default runs go to `./mlruns`. Start the UI with:

```bash
make mlflow-ui          # http://localhost:5000
```

To use a remote server, set `MLFLOW_TRACKING_URI` (an http URL or a SageMaker managed MLflow ARN).
Each model gets one run tagged with its family, GPU and torch version.

### 5. AWS SageMaker

This needs AWS credentials, an execution role with S3 and SageMaker access, and GPU quota for
`ml.g5.2xlarge` (or change `sagemaker.instance_type` in `configs/benchmark.yaml`).

```bash
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<SageMakerRole>
cabinsense sagemaker upload --export-dir data/sviro_export        # prints the s3:// URI
cabinsense sagemaker launch --data-uri s3://<bucket>/cabinsense/sviro_export [--only yolov8m]
```

Each job runs `cabinsense/cloud/entrypoint.py`, trains and evaluates one model, logs to
`MLFLOW_TRACKING_URI` when it is set, and saves `outputs/<name>/` in `model.tar.gz`.

### 6. Inference on your own frames

```bash
python scripts/infer.py --model configs/models/faster_rcnn.yaml \
    --checkpoint outputs/faster_rcnn/final.pt --vehicle aclass frame.png --out viz/
# {"image": "frame.png", "seats": {"left": "person", "middle": "empty", "right": "infant_seat"}, "child_present": true}
```

## How to test

The unit tests build tiny synthetic SVIRO-style trees (real file naming, real `.npy` box
format), so they need neither the dataset nor a GPU:

```bash
pip install -e ".[dev]"
make test            # or: python -m pytest -q
```

They cover file-name and box parsing, class mapping, vehicle hold-out splits, COCO and YOLO
export, the dataset round trip, flip and resize transforms, config inheritance and overrides,
pycocotools scoring (perfect predictions give 1.0 and empty predictions give 0.0), seat
assignment and occupancy metrics, the LR schedule, the registry, the tracker helpers and the
leaderboard rendering. For an end-to-end check on real data, run `make smoke` after
`cabinsense prepare`.

## Notes and design choices

- **Detection classes.** The detector learns four classes: infant seat, child seat, person and
  everyday object. SVIRO's "empty infant/child seat" codes are merged into their seat class,
  because for safety an unoccupied child seat still counts as a child-seat region.
- **Seat regions** are normalised rectangles in `configs/seats/default.yaml`. A detection is
  assigned to a seat when at least `min_overlap` of its area falls inside that seat. Tune the
  regions per vehicle if you add cabins whose geometry differs from the default.
- **Fair comparison.** All families are scored by the same pycocotools code against the same
  annotation file, using their own native training recipes (Ultralytics and YOLOX keep their
  own augmentation pipelines).
