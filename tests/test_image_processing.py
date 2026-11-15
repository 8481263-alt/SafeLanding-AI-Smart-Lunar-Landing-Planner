import unittest

import numpy as np
from PIL import Image

from modules.image_processing import digitize_surface_photo


class DigitizeSurfacePhotoTests(unittest.TestCase):
    def test_normalizes_lighting_and_preserves_crater_contrast(self):
        width, height = 240, 180
        x = np.linspace(65, 180, width, dtype=np.float32)
        pixels = np.broadcast_to(x, (height, width)).copy()
        yy, xx = np.ogrid[:height, :width]
        distance = np.hypot(xx - width / 2, yy - height / 2)
        pixels[distance < 24] = 35
        pixels[(distance >= 30) & (distance < 36)] = 235
        photo = Image.fromarray(np.repeat(pixels[:, :, None].astype(np.uint8), 3, axis=2))

        original, digitized = digitize_surface_photo(photo)
        before = np.asarray(original.convert("L"), dtype=np.float32)
        after = np.asarray(digitized.convert("L"), dtype=np.float32)
        outer_before = abs(float(before[20, 20] - before[20, -21]))
        outer_after = abs(float(after[20, 20] - after[20, -21]))
        crater_contrast = float(after[height // 2, width // 2]) - float(after[height // 2, width // 2 + 42])

        self.assertLess(outer_after, outer_before / 2)
        self.assertLess(crater_contrast, -20)
        self.assertEqual(digitized.mode, "RGB")
        self.assertEqual(digitized.size, original.size)

    def test_limits_preview_size_without_cropping(self):
        photo = Image.new("RGB", (2000, 1000), (120, 120, 120))

        original, digitized = digitize_surface_photo(photo)

        self.assertEqual(original.size, (1600, 800))
        self.assertEqual(digitized.size, original.size)


if __name__ == "__main__":
    unittest.main()
