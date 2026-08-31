"""
noise_utils.py
--------------
Utilities to:
  1. Add synthetic salt-and-pepper noise to a clean image (so you can
     test/demo your filter and prove it works, since you likely don't
     have a "ground truth clean vs already-noisy" pair handy).
  2. Measure how good the denoising is, using standard image-quality
     metrics (MSE and PSNR) that most CV mini-project rubrics expect.
"""

import numpy as np


def add_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05,
                               salt_ratio: float = 0.5, seed: int = None) -> np.ndarray:
    """
    Corrupts `image` with salt-and-pepper (impulse) noise.

    amount:     fraction of total pixels to corrupt (0.05 = 5%)
    salt_ratio: of the corrupted pixels, what fraction become "salt"
                (white, 255) vs "pepper" (black, 0). 0.5 = equal split.
    seed:       set for reproducible noise (useful for report screenshots)
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(amount * h * w)

    # Salt (white) pixels
    num_salt = int(num_pixels * salt_ratio)
    coords = [rng.integers(0, h, num_salt), rng.integers(0, w, num_salt)]
    noisy[coords[0], coords[1]] = 255

    # Pepper (black) pixels
    num_pepper = num_pixels - num_salt
    coords = [rng.integers(0, h, num_pepper), rng.integers(0, w, num_pepper)]
    noisy[coords[0], coords[1]] = 0

    return noisy


def mse(original: np.ndarray, processed: np.ndarray) -> float:
    """Mean Squared Error — lower is better (0 = identical images)."""
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    return float(np.mean((original - processed) ** 2))


def psnr(original: np.ndarray, processed: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in dB — higher is better.
    This is the standard metric examiners expect for "how well did
    your filter restore the image". Typical good denoising results
    land in the 25-40 dB range depending on noise level.
    """
    error = mse(original, processed)
    if error == 0:
        return float("inf")
    return 20 * np.log10(max_pixel) - 10 * np.log10(error)
