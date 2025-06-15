.PHONY: install install-all prepare smoke benchmark report mlflow-ui test lint

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

install-all:
	$(PY) -m pip install -e ".[all,dev]"
	$(PY) -m pip install git+https://github.com/Megvii-BaseDetection/YOLOX.git

prepare:
	cabinsense prepare --data configs/data/sviro.yaml

# One-epoch Faster R-CNN run on a slice of the data to sanity-check the pipeline.
smoke:
	cabinsense train --model configs/models/faster_rcnn.yaml --no-mlflow \
		--model-set train.epochs=1 train.max_iters_per_epoch=20 train.batch_size=2 \
		--set eval.limit=20 eval.latency_runs=0 output_dir=outputs/smoke

benchmark:
	cabinsense benchmark --bench configs/benchmark.yaml

report:
	cabinsense report --output-dir outputs

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	$(PY) -m pytest

lint:
	ruff check cabinsense tests scripts
