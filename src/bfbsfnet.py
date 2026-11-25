#!/usr/bin/env python3
"""
bfbsfnet.py

A simple "unrolled" implementation of BFBSF as a single-pass pipeline (BFBSFNet-style).
This is a prototype that applies the structure of an unrolled network where each block
produces a bias estimate (via bilateral-like smoothing + Bézier-like smoothing) and updates the latent image.

Note: For exact parity with your Algo2 implementation, this file imports two helper functions
from Algo2.py if present:
  - bilateral_filter_img(z, sigma_space, sigma_color)
  - fit_bezier_surface(img, degree_u, degree_v, downsample, ridge)
If Algo2.py is not available, the code will still run using a Gaussian-based fallback.

Usage:
    python bfbsfnet.py
"""
from typing import List, Dict, Tuple
import numpy as np
import os

# Try to use helpers in Algo2 if available (your existing implementation).
try:
    from Algo2 import bilateral_filter_img, fit_bezier_surface
except Exception:
    bilateral_filter_img = None
    fit_bezier_surface = None
    from scipy.ndimage import gaussian_filter

    def bilateral_filter_img(img, sigma_space=8.0, sigma_color=0.05):
        # fallback: gaussian blur (not a bilateral filter, but safe)
        return gaussian_filter(img, sigma=sigma_space)

    def fit_bezier_surface(img, degree_u=4, degree_v=4, downsample=8, ridge=1e-4):
        # fallback: coarse bicubic-like smoothing via Gaussian on a downsampled grid
        small = img.copy()
        # coarse smoothing then upsample (naive)
        from skimage.transform import resize
        h, w = img.shape
        sh, sw = max(8, h // downsample), max(8, w // downsample)
        small = resize(img, (sh, sw), order=3, anti_aliasing=True, preserve_range=True)
        small = gaussian_filter(small, sigma=1.5)
        up = resize(small, (h, w), order=3, anti_aliasing=True, preserve_range=True)
        return up.astype(np.float32)


def BFBSFNet_unrolled(z: np.ndarray, params: List[Dict]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run unrolled BFBSF-like pipeline.
    z: degraded image float32 in [0,1]
    params: list of dicts with keys: sigma_space, sigma_color, bezier_degree, gamma
    Returns: corrected image f_final and list of biases (smoothed) per block.
    """
    f = z.copy().astype(np.float32)
    biases = []
    for i, p in enumerate(params):
        # bilateral-like block (if Algo2 provides bilateral, it will be used)
        bf = bilateral_filter_img(z, sigma_space=float(p.get('sigma_space', 8.0)),
                                      sigma_color=float(p.get('sigma_color', 0.05)))
        # Bezier-like smoothing to enforce low-frequency bias
        deg_u, deg_v = p.get('bezier_degree', (4, 4))
        b_bezier = fit_bezier_surface(bf, degree_u=deg_u, degree_v=deg_v,
                                      downsample=p.get('downsample', 8), ridge=p.get('ridge', 1e-4))
        biases.append(b_bezier.astype(np.float32))
        # latent update (mimics subtraction step in BFBSF)
        gamma = float(p.get('gamma', 0.3))
        f = f - gamma * b_bezier
    return f, biases


def default_params(n_blocks: int = 3):
    """Return default per-block parameters (intended as starting point for experiments)."""
    params = []
    # coarse-to-fine bilateral radiometric/spatial scales
    sigma_space_list = [16, 8, 4][:n_blocks]
    sigma_color_list = [0.12, 0.05, 0.02][:n_blocks]
    bezier_degs = [(4, 4), (4, 4), (3, 3)][:n_blocks]
    for i in range(n_blocks):
        params.append({
            'sigma_space': sigma_space_list[i],
            'sigma_color': sigma_color_list[i],
            'bezier_degree': bezier_degs[i],
            'downsample': 8,
            'ridge': 1e-4,
            'gamma': 0.3
        })
    return params


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float
    import numpy as np
    import cv2


    path = "output_degraded.png"
    if not os.path.exists(path):
        # synthetic test pattern (coarse bias + small textures)
        I = np.zeros((256, 256), dtype=np.float32)
        cx, cy = 64, 64
        xv, yv = np.meshgrid(np.arange(256), np.arange(256))
        r2 = (xv - cx) ** 2 + (yv - cy) ** 2
        I += 0.5 * np.exp(-r2 / (2 * (24 ** 2)))
        I += 0.3 * (np.sin(xv / 10.0) * np.cos(yv / 14.0)).astype(np.float32)
        I = np.clip(I, 0.0, 1.0)
    else:
        I = img_as_float(io.imread(path, as_gray=True)).astype(np.float32)
        I = np.clip(I, 0, 1)

    params = default_params(3)
    f_out, biases = BFBSFNet_unrolled(I, params)
    # Scale to 0–255 and convert to uint8 before saving
    f_out_uint8 = (np.clip(f_out, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite("bfbsfnet_output.png", f_out_uint8)

    print("✅ Output image saved as bfbsfnet_output.png")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.title("Original degraded"); plt.imshow(I, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("BFBSFNet (unrolled)"); plt.imshow(np.clip(f_out, 0, 1), cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("First bias"); plt.imshow(biases[0], cmap='gray'); plt.axis('off')
    plt.show()
