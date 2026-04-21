from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_preprocessed_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        resized = resize_shortest_side(rgb, image_size)
        cropped = center_crop(resized, image_size)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def resize_shortest_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        msg = f"Invalid image dimensions: {image.size}"
        raise ValueError(msg)
    scale = target_size / min(width, height)
    resized = (round(width * scale), round(height * scale))
    return image.resize(resized, Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    return image.crop((left, top, left + crop_size, top + crop_size))
