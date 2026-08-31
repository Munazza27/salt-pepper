"""
median_filter.py
-----------------
A from-scratch implementation of the Median Filter for removing
Salt-and-Pepper (impulse) noise from images.

Why median filtering works for salt-and-pepper noise:
    Salt-and-pepper noise randomly replaces some pixels with pure
    black (0) or pure white (255). These are extreme outlier values.
    A MEAN filter would blend the outlier into the average, spreading
    its damage into neighboring pixels (and the image gets blurry).
    A MEDIAN filter instead picks the *middle* value of a sorted
    neighborhood. Since noisy pixels are extreme (very high/very low),
    they get pushed to the ends of the sorted list and are almost
    never picked as the median -- so they get replaced with a
    realistic value from the true image, while edges and details in
    the untouched pixels are preserved much better than with mean
    filtering.

This file implements the filter manually (no cv2.medianBlur) so you
can explain every step in a viva.
"""

import numpy as np


def pad_image(image: np.ndarray, pad_size: int, mode: str = "edge") -> np.ndarray:
    """
    Pads a 2D (grayscale) or 3D (color) image so that the sliding
    window can be centered even on border pixels.

    mode="edge" repeats the edge pixels outward (this avoids
    introducing fake zeros/black borders that would otherwise get
    treated as extra "pepper" noise near the edges).
    """
    if image.ndim == 2:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    else:
        pad_width = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    return np.pad(image, pad_width=pad_width, mode=mode)


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies a median filter to `image` using a square kernel of size
    kernel_size x kernel_size (must be odd: 3, 5, 7, ...).

    Works for both grayscale (H, W) and color (H, W, C) images —
    each channel is filtered independently.

    Steps (classic sliding-window algorithm):
      1. Pad the image so edge pixels have a full neighborhood.
      2. For every pixel, take the kernel_size x kernel_size
         neighborhood around it.
      3. Sort the neighborhood values and pick the middle (median)
         value.
      4. Place that median value at the corresponding pixel in the
         output image.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be an odd number >= 3 (e.g. 3, 5, 7)")

    pad = kernel_size // 2
    padded = pad_image(image, pad)
    output = np.zeros_like(image)

    if image.ndim == 2:
        h, w = image.shape
        for i in range(h):
            for j in range(w):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.median(window)
    else:
        h, w, c = image.shape
        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    window = padded[i:i + kernel_size, j:j + kernel_size, ch]
                    output[i, j, ch] = np.median(window)

    return output.astype(image.dtype)


def median_filter_fast(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    A vectorized version of the same algorithm using numpy's stride
    tricks, useful for larger images where the pure Python double
    loop above would be too slow. Produces IDENTICAL results to
    median_filter() -- same concept, just faster execution.

    Good to mention in your report: "we also optimized the naive
    O(H*W*k^2 log(k^2)) loop using vectorization for practical use
    on larger images."
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("kernel_size must be an odd number >= 3 (e.g. 3, 5, 7)")

    pad = kernel_size // 2

    if image.ndim == 2:
        padded = pad_image(image, pad)
        windows = sliding_window_view(padded, (kernel_size, kernel_size))
        result = np.median(windows, axis=(-2, -1))
        return result.astype(image.dtype)
    else:
        channels = []
        for ch in range(image.shape[2]):
            padded = pad_image(image[:, :, ch], pad)
            windows = sliding_window_view(padded, (kernel_size, kernel_size))
            channels.append(np.median(windows, axis=(-2, -1)))
        return np.stack(channels, axis=-1).astype(image.dtype)
