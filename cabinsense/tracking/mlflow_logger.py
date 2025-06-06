"""Thin MLflow wrapper so training code never imports mlflow directly.

The tracking server is taken from ``MLFLOW_TRACKING_URI`` (defaults to a local ``./mlruns``
store). ``NullTracker`` keeps the same interface for dry runs and unit tests.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

log = logging.getLogger(__name__)

_INVALID_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-./ ]")


def flatten(d: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        full = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten(value, full))
        else:
            out[full] = value
    return out


def sanitize_key(key: str) -> str:
    return _INVALID_KEY_CHARS.sub("_", key)


class NullTracker:
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log_params(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str | Path, artifact_path: Optional[str] = None) -> None:
        pass

    def log_dict(self, payload: Dict, name: str) -> None:
        pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        pass

    @property
    def run_id(self) -> Optional[str]:
        return None


class MLflowTracker(NullTracker):
    def __init__(self, experiment: str, tracking_uri: Optional[str] = None) -> None:
        import mlflow

        self._mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self._run = None

    @contextmanager
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None) -> Iterator["MLflowTracker"]:
        self._run = self._mlflow.start_run(run_name=run_name, tags=tags)
        try:
            yield self
        except Exception:
            self._mlflow.end_run(status="FAILED")
            self._run = None
            raise
        else:
            self._mlflow.end_run(status="FINISHED")
            self._run = None

    @property
    def run_id(self) -> Optional[str]:
        return self._run.info.run_id if self._run else None

    def log_params(self, params: Mapping[str, Any]) -> None:
        flat = {sanitize_key(k): str(v)[:500] for k, v in flatten(params).items()}
        items = list(flat.items())
        for i in range(0, len(items), 100):  # MLflow caps batch size
            self._mlflow.log_params(dict(items[i : i + 100]))

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isnan(value) or math.isinf(value):
                continue
            clean[sanitize_key(key)] = float(value)
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, artifact_path: Optional[str] = None) -> None:
        path = Path(path)
        if not path.exists():
            log.warning("Artifact %s does not exist; skipping", path)
            return
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, payload: Dict, name: str) -> None:
        self._mlflow.log_text(json.dumps(payload, indent=2, default=str), name)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._mlflow.set_tags(dict(tags))


def build_tracker(cfg: Mapping[str, Any]):
    tcfg = cfg.get("tracking", {})
    if not tcfg.get("enabled", True):
        return NullTracker()
    return MLflowTracker(tcfg.get("experiment", cfg.get("experiment", "cabinsense")), tcfg.get("tracking_uri") or None)
