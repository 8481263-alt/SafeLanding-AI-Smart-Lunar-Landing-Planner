"""Image preparation for surface-photo review and hazard detection."""

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage

MAX_SURFACE_SIDE = 1600


def digitize_surface_photo(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Return a resized source preview and a grayscale, lighting-normalized image."""
    source = ImageOps.exif_transpose(image).convert("RGB")
    source.thumbnail((MAX_SURFACE_SIDE, MAX_SURFACE_SIDE), Image.Resampling.LANCZOS)

    gray = source.convert("L").filter(ImageFilter.MedianFilter(size=3))
    pixels = np.asarray(gray, dtype=np.float32)
    sigma = max(12.0, min(pixels.shape) / 8)
    background = ndimage.gaussian_filter(pixels, sigma=sigma, mode="reflect")
    flattened = pixels - background + float(np.median(background))
    low, high = np.percentile(flattened, (1, 99))

    if high - low > 1:
        enhanced = np.clip((flattened - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
        digitized = Image.fromarray(enhanced)
    else:
        digitized = gray

    digitized = digitized.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=4))
    return source, digitized.convert("RGB")
