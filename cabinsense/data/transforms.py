"""Detection-aware transforms operating on ``(PIL.Image, target dict)`` pairs.

Targets carry ``boxes`` as an ``(N, 4)`` float tensor in absolute xyxy pixels and ``labels``
as an ``(N,)`` int64 tensor. Geometric transforms update boxes and record the applied scale so
predictions can be mapped back to original image coordinates.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToRGB:
    """SVIRO grayscale/IR frames are single channel; detectors expect three."""

    def __call__(self, image: Image.Image, target):
        return image.convert("RGB"), target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target):
        if random.random() >= self.p:
            return image, target
        width = image.width
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target["boxes"] = boxes
        if "seat_labels" in target:
            # Left and right seats swap under a horizontal flip.
            target["seat_labels"] = target["seat_labels"].flip(0)
        return image, target


class ColorJitter:
    def __init__(self, brightness: float = 0.3, contrast: float = 0.3, p: float = 0.8):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image: Image.Image, target):
        if random.random() < self.p:
            image = F.adjust_brightness(image, 1 + random.uniform(-self.brightness, self.brightness))
            image = F.adjust_contrast(image, 1 + random.uniform(-self.contrast, self.contrast))
        return image, target


class ResizeLongestSide:
    """Resize so the longest side equals ``size`` and pad to ``size x size`` (top-left anchored)."""

    def __init__(self, size: int, pad: bool = True):
        self.size = size
        self.pad = pad

    def __call__(self, image: Image.Image, target):
        w, h = image.size
        scale = self.size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        if self.pad:
            canvas = Image.new(image.mode, (self.size, self.size))
            canvas.paste(image, (0, 0))
            image = canvas
        if target["boxes"].numel():
            target["boxes"] = target["boxes"] * scale
        target["scale"] = torch.tensor(scale, dtype=torch.float32)
        return image, target


class ToTensor:
    def __call__(self, image: Image.Image, target):
        return F.to_tensor(image), target


class Normalize:
    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406), std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, image: torch.Tensor, target):
        return F.normalize(image, self.mean, self.std), target


def build_transforms(train: bool, image_size: Optional[int] = None, normalize: bool = False) -> Compose:
    """Default pipeline.

    torchvision detectors normalise and resize internally, so they use ``image_size=None``;
    EfficientDet needs a fixed square, normalised input.
    """
    ts: List[Callable] = [ToRGB()]
    if train:
        ts += [RandomHorizontalFlip(0.5), ColorJitter()]
    if image_size:
        ts.append(ResizeLongestSide(image_size))
    ts.append(ToTensor())
    if normalize:
        ts.append(Normalize())
    return Compose(ts)
