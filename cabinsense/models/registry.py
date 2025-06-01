"""Name -> detector class registry so configs can pick a family by string."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Optional, Type

from .base import Detector

_REGISTRY: Dict[str, Type[Detector]] = {}

# Families whose optional dependencies are imported lazily on first use.
_LAZY_MODULES = {
    "torchvision": "cabinsense.models.torchvision_models",
    "efficientdet": "cabinsense.models.efficientdet",
    "yolov8": "cabinsense.models.yolov8",
    "yolox": "cabinsense.models.yolox_model",
}


def register(family: str) -> Callable[[Type[Detector]], Type[Detector]]:
    def wrap(cls: Type[Detector]) -> Type[Detector]:
        if family in _REGISTRY and _REGISTRY[family] is not cls:
            raise KeyError(f"Detector family {family!r} already registered")
        cls.family = family
        _REGISTRY[family] = cls
        return cls

    return wrap


def get_detector_class(family: str) -> Type[Detector]:
    if family not in _REGISTRY and family in _LAZY_MODULES:
        importlib.import_module(_LAZY_MODULES[family])
    try:
        return _REGISTRY[family]
    except KeyError as exc:
        known = sorted(set(_REGISTRY) | set(_LAZY_MODULES))
        raise KeyError(f"Unknown detector family {family!r}; known: {known}") from exc


def build_detector(model_cfg: Dict[str, Any], num_classes: int, device: Optional[str] = None) -> Detector:
    family = model_cfg["family"]
    return get_detector_class(family)(model_cfg, num_classes=num_classes, device=device)


def available_families() -> list:
    return sorted(set(_REGISTRY) | set(_LAZY_MODULES))
