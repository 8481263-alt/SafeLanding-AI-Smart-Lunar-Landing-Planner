import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from train_hazard import HazardNet  # файл від Людини 1, лежить у корені проєкту

ROOT = Path(__file__).resolve().parent.parent
_model = None
_size = None

def _load():
    """Модель вантажиться один раз, а не на кожен запит."""
    global _model, _size
    if _model is None:
        ckpt = torch.load(ROOT / "training/best.pt", map_location="cpu", weights_only=False)
        model = HazardNet(pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        _model, _size = model, ckpt["config"]["size"]
    return _model, _size

def prepare_input(image: Image.Image, size: int) -> torch.Tensor:
    """Готує вхід [1, 3, size, size] так само, як настільне демо (код Людини 1)."""
    gray_rgb = image.convert("RGB").convert("L").convert("RGB")
    scale = min(size / image.width, size / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = gray_rgb.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - width) // 2, (size - height) // 2))

    pixels = np.array(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (pixels - mean) / std
    return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)

def predict(image_bytes: bytes, threshold: float = 0.5, min_area: int = 30) -> dict:
    model, size = _load()
    t0 = time.perf_counter()

    with Image.open(io.BytesIO(image_bytes)) as image:
        w, h = image.size
        x = prepare_input(image, size)

    with torch.inference_mode():
        hazard = model(x).sigmoid()[0, 0].numpy()  # [size, size], 0..1

    # Параметри letterbox: повертаємо координати в оригінальне зображення
    scale = min(size / w, size / h)
    ox = (size - max(1, round(w * scale))) // 2
    oy = (size - max(1, round(h * scale))) // 2

    labels, _ = ndimage.label(hazard > threshold)
    hazards = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels[sl] == i
        if mask.sum() < min_area:
            continue
        ys, xs = sl
        hazards.append({
            "bbox": [
                max(0.0, (xs.start - ox) / scale),
                max(0.0, (ys.start - oy) / scale),
                min(float(w), (xs.stop - ox) / scale),
                min(float(h), (ys.stop - oy) / scale),
            ],
            "cls": "hazard",
            "confidence": float(hazard[sl][mask].mean()),
        })

    return {
        "hazards": hazards,
        "image_width": w,
        "image_height": h,
        "inference_ms": (time.perf_counter() - t0) * 1000,
        "device": "cpu",
    }