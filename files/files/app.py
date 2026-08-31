"""
app.py
------
Streamlit GUI for the Salt-and-Pepper Noise Reduction System.

Run locally with:
    pip install streamlit numpy pillow
    streamlit run app.py

Then open the local URL it prints (usually http://localhost:8501).

Workflow:
  1. Upload an image.
  2. (Optional) Add synthetic salt-and-pepper noise to it, so you can
     demo the "before vs after" even with a clean input image.
  3. Apply the from-scratch median filter.
  4. View original / noisy / denoised side by side, with PSNR shown.
  5. Download the cleaned image.
"""

import io
import numpy as np
import streamlit as st
from PIL import Image

from median_filter import median_filter_fast
from noise_utils import add_salt_and_pepper_noise, psnr, mse

st.set_page_config(page_title="Salt & Pepper Noise Reduction", layout="wide")

st.title("🧂 Salt-and-Pepper Noise Reduction System")
st.caption("Median filtering, implemented from scratch — Computer Vision mini project")

with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

    st.subheader("1. Add test noise (optional)")
    inject_noise = st.checkbox("Add synthetic salt-and-pepper noise", value=True,
                                help="Turn this off if your uploaded image is already noisy.")
    noise_amount = st.slider("Noise amount", 0.01, 0.30, 0.05, 0.01)

    st.subheader("2. Filter settings")
    kernel_size = st.select_slider("Kernel size", options=[3, 5, 7, 9], value=3)

if uploaded_file is not None:
    original = np.array(Image.open(uploaded_file).convert("RGB"))

    if inject_noise:
        working_image = add_salt_and_pepper_noise(original, amount=noise_amount, seed=42)
        stage_label = "Noisy Input (synthetic)"
    else:
        working_image = original
        stage_label = "Uploaded Input"

    with st.spinner("Applying median filter..."):
        denoised = median_filter_fast(working_image, kernel_size=kernel_size)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original, caption="Original", use_container_width=True)
    with col2:
        st.image(working_image, caption=stage_label, use_container_width=True)
    with col3:
        st.image(denoised, caption=f"Denoised (median, k={kernel_size})", use_container_width=True)

    st.subheader("Quality metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PSNR before filtering", f"{psnr(original, working_image):.2f} dB")
    m2.metric("PSNR after filtering", f"{psnr(original, denoised):.2f} dB")
    m3.metric("MSE before filtering", f"{mse(original, working_image):.1f}")
    m4.metric("MSE after filtering", f"{mse(original, denoised):.1f}")
    st.caption("PSNR: higher is better. MSE: lower is better. "
               "(These are only meaningful when synthetic noise was added, "
               "since we need the clean original to compare against.)")

    # Download button
    result_img = Image.fromarray(denoised)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button("Download denoised image", data=buf.getvalue(),
                        file_name="denoised.png", mime="image/png")
else:
    st.info("👈 Upload an image from the sidebar to get started.")
