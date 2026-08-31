# Salt-and-Pepper Noise Reduction System
### Computer Vision Mini Project — Median Filtering

## 1. Problem Statement
Salt-and-pepper noise (also called impulse noise) randomly corrupts individual
pixels in an image, setting them to pure black (0) or pure white (255).
It commonly arises from faulty camera sensors, transmission errors, or bit
errors in storage. This project builds a system that removes this noise from
uploaded images using **median filtering**.

## 2. Why Median Filtering?
| Filter Type | Behavior on Salt & Pepper Noise |
|---|---|
| Mean (average) filter | Blends the extreme noisy pixel into its neighbors → noise spreads instead of disappearing; image gets blurry |
| Gaussian filter | Same weakness as mean filtering — smooths but doesn't remove outliers |
| **Median filter** | Replaces each pixel with the **median** of its neighborhood. Since noisy pixels are extreme outliers (0 or 255), they get sorted to the ends and are almost never chosen as the median — so they're effectively removed while edges are preserved |

This is the key concept to explain in your viva: median filtering is a
**non-linear, rank-order filter** that is robust to outliers, which is exactly
what salt-and-pepper noise is.

## 3. Algorithm (from scratch)
1. **Pad** the image borders (edge-replication padding) so a full
   neighborhood exists even for border pixels.
2. **Slide a k×k window** (k = 3, 5, 7...) across every pixel.
3. **Sort** the pixel values inside the window.
4. **Pick the middle value** (the median) and assign it to the output pixel.
5. Repeat for every channel (R, G, B) independently for color images.

Time complexity: O(H × W × k² log(k²)) for the naive version (sorting each
window). The project includes a vectorized version
(`median_filter_fast`) using numpy's `sliding_window_view` for practical
speed on larger images — same algorithm, same output, just faster.

## 4. Project Structure
```
sp_noise_project/
├── median_filter.py   # Core algorithm (from scratch) — median_filter(), median_filter_fast()
├── noise_utils.py      # add_salt_and_pepper_noise(), mse(), psnr()
├── app.py               # Streamlit GUI — upload an image, denoise it, download result
└── README.md
```

## 5. How to Run the GUI
```bash
pip install streamlit numpy pillow
streamlit run app.py
```
This opens a browser window where you can:
- Upload any image
- Optionally inject synthetic salt-and-pepper noise (useful for demoing,
  since you can then compare against the known-clean original)
- Choose the kernel size (3×3, 5×5, 7×7, 9×9)
- View original / noisy / denoised side-by-side
- See PSNR and MSE quality metrics
- Download the cleaned image

## 6. Evaluation Metrics
- **MSE (Mean Squared Error):** average squared pixel difference from the
  clean original. Lower = better.
- **PSNR (Peak Signal-to-Noise Ratio, dB):** standard image-quality metric.
  Higher = better. In our demo test image, PSNR improved from **~15.4 dB
  (noisy)** to **~26.4 dB (denoised, k=3)**.

## 7. Kernel Size Trade-off
- **Smaller kernel (3×3):** removes noise while keeping fine details/edges
  sharper. Best for light-to-moderate noise.
- **Larger kernel (5×5, 7×7):** removes heavier noise more thoroughly but
  starts to blur fine details and thin lines/text.
- This trade-off (noise suppression vs. detail preservation) is a good point
  to discuss in your report/viva.

## 8. Possible Extensions (if you want to go further)
- **Adaptive median filter**: increases kernel size dynamically only in
  regions still detected as noisy, preserving detail elsewhere.
- Compare against `cv2.medianBlur()` to validate your from-scratch
  implementation matches the industry-standard result.
- Test on real noisy images (e.g., old scanned documents, transmission-
  corrupted images) instead of only synthetic noise.
- Extend the GUI to support batch processing of multiple images.

## 9. Likely Viva Questions
- **Q: Why not use a mean/Gaussian filter instead?**
  A: Mean/Gaussian filters average in the noisy pixel's extreme value,
  spreading the error. Median filtering discards it because it's rarely the
  middle value of a sorted window.
- **Q: What happens with even kernel sizes?**
  A: There's no single middle element, so kernel size must be odd (3, 5, 7...).
- **Q: What's the effect of increasing kernel size?**
  A: More noise removed, but more blurring/loss of fine detail — a
  classic trade-off.
- **Q: How would you measure how well the filter worked?**
  A: PSNR and MSE against a known-clean reference image.
