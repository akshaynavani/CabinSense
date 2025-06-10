"""Launch CabinSense training jobs on AWS SageMaker.

Each model in the benchmark becomes its own PyTorch training job on a GPU instance. The job
receives the exported SVIRO dataset as the ``sviro`` input channel and runs
:mod:`cabinsense.cloud.entrypoint`, which reuses :func:`cabinsense.benchmark.run_model`.

Required environment (see ``.env.example``):
    SAGEMAKER_ROLE_ARN   IAM role the job assumes
    SAGEMAKER_BUCKET     S3 bucket for data, code and outputs (defaults to the session bucket)
    AWS_REGION / AWS_PROFILE as usual for boto3
Optional:
    MLFLOW_TRACKING_URI  remote MLflow server (e.g. SageMaker managed MLflow ARN or http URL)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import load_config

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    "instance_type": "ml.g5.2xlarge",
    "instance_count": 1,
    "framework_version": "2.2",
    "py_version": "py310",
    "volume_size": 100,
    "max_run_hours": 24,
    "use_spot": False,
    "max_wait_hours": 36,
}


def _session(region: Optional[str] = None):
    import boto3
    import sagemaker

    boto_sess = boto3.Session(region_name=region or os.environ.get("AWS_REGION"))
    return sagemaker.Session(boto_session=boto_sess)


def upload_dataset(export_dir: Path, prefix: str = "cabinsense/sviro_export", bucket: Optional[str] = None) -> str:
    """Upload an exported dataset (prepared with ``--copy``, not symlinks) and return its S3 URI."""
    sess = _session()
    bucket = bucket or os.environ.get("SAGEMAKER_BUCKET") or sess.default_bucket()
    uri = sess.upload_data(str(export_dir), bucket=bucket, key_prefix=prefix)
    log.info("Uploaded %s -> %s", export_dir, uri)
    return uri


def _stage_source() -> Path:
    """Copy only code + configs into a temp dir so local data/outputs never get uploaded."""
    stage = Path(tempfile.mkdtemp(prefix="cabinsense_src_"))
    shutil.copytree(PROJECT_ROOT / "cabinsense", stage / "cabinsense", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT_ROOT / "configs", stage / "configs")
    # SageMaker pip-installs requirements.txt found in source_dir. YOLOX needs a git install.
    reqs = (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    reqs = [r for r in reqs if r and not r.startswith("#") and not r.startswith("torch")]
    reqs.append("yolox @ git+https://github.com/Megvii-BaseDetection/YOLOX.git")
    (stage / "requirements.txt").write_text("\n".join(reqs) + "\n")
    shutil.copy2(Path(__file__).with_name("entrypoint.py"), stage / "train.py")
    return stage


def launch_job(
    model_config: str,
    data_uri: str,
    bench_config: str = "configs/benchmark.yaml",
    sm_cfg: Optional[Dict[str, Any]] = None,
    overrides: Optional[List[str]] = None,
    wait: bool = False,
):
    from sagemaker.pytorch import PyTorch

    sm = {**DEFAULTS, **(sm_cfg or {})}
    role = sm.get("role") or os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        raise EnvironmentError("Set SAGEMAKER_ROLE_ARN or sagemaker.role in the benchmark config")
    sess = _session(sm.get("region"))
    bucket = sm.get("bucket") or os.environ.get("SAGEMAKER_BUCKET") or sess.default_bucket()
    model_name = load_config(PROJECT_ROOT / model_config).get("name", Path(model_config).stem)

    env = {"PYTHONUNBUFFERED": "1"}
    for key in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "YOLOX_WEIGHTS"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    hyperparameters = {"model-config": model_config, "bench-config": bench_config}
    if overrides:
        hyperparameters["overrides"] = " ".join(overrides)

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(_stage_source()),
        role=role,
        framework_version=sm["framework_version"],
        py_version=sm["py_version"],
        instance_type=sm["instance_type"],
        instance_count=int(sm["instance_count"]),
        volume_size=int(sm["volume_size"]),
        max_run=int(float(sm["max_run_hours"]) * 3600),
        use_spot_instances=bool(sm["use_spot"]),
        max_wait=int(float(sm["max_wait_hours"]) * 3600) if sm["use_spot"] else None,
        checkpoint_s3_uri=f"s3://{bucket}/cabinsense/checkpoints/{model_name}" if sm["use_spot"] else None,
        output_path=f"s3://{bucket}/cabinsense/outputs",
        base_job_name=f"cabinsense-{model_name}".replace("_", "-")[:32],
        hyperparameters=hyperparameters,
        environment=env,
        sagemaker_session=sess,
        disable_profiler=True,
    )
    estimator.fit({"sviro": data_uri}, wait=wait)
    job = estimator.latest_training_job.name
    log.info("Launched SageMaker job %s for %s", job, model_name)
    return job


def launch_benchmark(bench_config: str, data_uri: str, only: Optional[List[str]] = None, wait: bool = False) -> List[str]:
    bench = load_config(PROJECT_ROOT / bench_config)
    jobs = []
    for model_cfg in bench["models"]:
        name = load_config(PROJECT_ROOT / model_cfg).get("name")
        if only and name not in only:
            continue
        jobs.append(launch_job(model_cfg, data_uri, bench_config, bench.get("sagemaker"), wait=wait))
    return jobs
