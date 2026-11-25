#!/usr/bin/env python3
"""
pabr.py

Physics-Aware Bias Recovery (PABR) prototype.
Minimizes:
  J(b) = 0.5 * || z - f - b ||_2^2  + 0.5 * lambda_pde * || Lap(b) + h * (b - T_air) ||_2^2  + 0.5 * lambda_smooth * ||grad b||_2^2

This prototype uses simple finite-difference Laplacian and gradient-descent optimization to find b.
It aims to demonstrate the physics-informed penalty (PDE residual) in the optimization.

Usage:
    python pabr.py
"""
from typing import Tuple, Dict
import numpy as np
from scipy.ndimage import laplace, gaussian_filter
import os

def pabr_correct(z: np.ndarray, f_init: np.ndarray, params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    z: degraded image
    f_init: initial latent image estimate (can be z or a prior)
    params: dictionary of hyperparameters
    Returns: corrected image f_corr and estimated bias b_hat
    """
    z = z.astype(np.float32)
    f = f_init.astype(np.float32)
    # initialize bias estimate using bilateral-like smoothing (falls back to gaussian)
    sigma_space = params.get('sigma_space', 8.0)
    try:
        from Algo2 import bilateral_filter_img
        b = bilateral_filter_img(z, sigma_space=sigma_space, sigma_color=params.get('sigma_color', 0.05)).astype(np.float32)
    except Exception:
        b = gaussian_filter(z, sigma=sigma_space).astype(np.float32)

    lambda_pde = float(params.get('lambda_pde', 1.0))
    h = float(params.get('h', 0.02))
    T_air = float(params.get('T_air', 0.0))
    lambda_smooth = float(params.get('lambda_smooth', 0.01))
    lr = float(params.get('lr', 0.2))
    n_iter = int(params.get('n_iter', 200))

    # gradient descent on b
    for it in range(n_iter):
        # data residual derivative wrt b
        data_res = (b + f) - z  # gradient: b+f - z
        # PDE residual: Lap(b) + h*(b - T_air)
        lapb = laplace(b)
        pde_res = lapb + h * (b - T_air)
        # gradient of PDE penalty wrt b approximated by Lap^T(pde_res) + h * pde_res
        # since discrete Laplacian is symmetric (approx), Lap^T == Lap
        grad_pde = 2.0 * lambda_pde * (laplace(pde_res) + h * pde_res)
        # smoothness gradient approximated using Laplacian (negative divergence of gradient)
        grad_smooth = 2.0 * lambda_smooth * (-lapb)
        # total gradient
        grad_total = 2.0 * data_res + grad_pde + grad_smooth

        # update
        b = b - lr * grad_total
        # keep b in reasonable numeric range to avoid blow-up in practice
        b = np.clip(b, -1.0, 1.0)

        # adaptive LR schedule (simple)
        if (it + 1) % 50 == 0:
            lr *= 0.8

    f_corr = f - b
    return f_corr, b


def default_params():
    return {
        'sigma_space': 8.0,
        'sigma_color': 0.05,
        'lambda_pde': 5.0,
        'h': 0.02,
        'T_air': 0.0,
        'lambda_smooth': 0.01,
        'lr': 0.2,
        'n_iter': 200
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float
    import numpy as np

    path = "output_degraded.png"
    if not os.path.exists(path):
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

    f_init = I.copy()
    params = default_params()
    f_corr, b_hat = pabr_correct(I, f_init, params)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1); plt.title("Original degraded"); plt.imshow(I, cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("PABR corrected"); plt.imshow(np.clip(f_corr, 0, 1), cmap='gray'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Estimated bias (b_hat)"); plt.imshow(b_hat, cmap='gray'); plt.axis('off')
    plt.show()
