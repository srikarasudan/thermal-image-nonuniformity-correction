import cupy as cp
import numpy as np
import cv2
import torch
from skimage.feature import hog
from tqdm import trange

# -------------------------
# Utility functions
# -------------------------
def bilateral_filter_img(img, sigma_space, sigma_color, diameter=None):
    """
    Wrapper for OpenCV bilateral filter (CPU).
    img: 2D float image in range [0,1] or [0,255] (on CPU)
    """
    img_cpu = cp.asnumpy(img) if isinstance(img, cp.ndarray) else img
    if img_cpu.dtype != np.uint8:
        img8 = np.clip(img_cpu * 255.0, 0, 255).astype(np.uint8)
    else:
        img8 = img_cpu
    if diameter is None:
        diameter = int(2 * np.ceil(2 * sigma_space) + 1)
        if diameter <= 0:
            diameter = 3
    res = cv2.bilateralFilter(img8, diameter, sigma_color, sigma_space)
    if img_cpu.dtype != np.uint8:
        return cp.array(res.astype(np.float32) / 255.0)
    else:
        return cp.array(res)

def gradient_and_orientation(img):
    """
    Returns gradient magnitude and orientation (radians in [0, 2pi)).
    Uses PyTorch for GPU gradients.
    img is 2D float (CuPy array).
    """
    img_torch = torch.from_numpy(cp.asnumpy(img)).float().cuda()
    gx = torch.nn.functional.conv2d(img_torch[None, None], torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device='cuda'), padding=1)[0,0]
    gy = torch.nn.functional.conv2d(img_torch[None, None], torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device='cuda'), padding=1)[0,0]
    mag = torch.hypot(gx, gy)
    ori = torch.atan2(gy, gx)
    ori = torch.remainder(ori, 2 * np.pi)
    return mag.cpu().numpy(), ori.cpu().numpy()

# -------------------------
# Regularizers
# -------------------------
def R_bias(bias, radiation_center=None):
    h, w = bias.shape
    mag, ori = gradient_and_orientation(bias)
    l1_grad = cp.sum(cp.abs(cp.array(mag)))

    if radiation_center is None:
        cx, cy = (w-1)/2.0, (h-1)/2.0
    else:
        cx, cy = radiation_center

    ys, xs = cp.indices(bias.shape)
    vec_x = cx - xs
    vec_y = cy - ys
    theta_center = cp.mod(cp.arctan2(vec_y, vec_x), 2*cp.pi)
    diff = cp.abs(cp.angle(cp.exp(1j*(cp.array(ori) - theta_center))))
    aligned_mask = (diff <= (cp.pi/4.0)).astype(cp.float32)
    orientation_count = cp.sum(aligned_mask * (cp.array(mag) > 1e-6))
    return float(l1_grad + orientation_count)

def R_image_hog(img, n_cells=(9,9)):
    h, w = img.shape
    cx, cy = w//2, h//2
    blocks = [
        (0, cy, 0, cx),
        (0, cy, cx, w),
        (cy, h, cx, w),
        (cy, h, 0, cx),
    ]
    total = 0.0
    img_cpu = cp.asnumpy(img)
    for (y0,y1,x0,x1) in blocks:
        patch = img_cpu[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        hog_vec = hog(patch, pixels_per_cell=(max(1, patch.shape[0]//n_cells[0]), max(1, patch.shape[1]//n_cells[1])),
                      cells_per_block=(1,1), orientations=9, feature_vector=True)
        if hog_vec.size == 0:
            continue
        mean_vec = np.mean(hog_vec)
        total += np.sum((hog_vec - mean_vec)**2)
    return total

# -------------------------
# BÃ©zier surface fitting
# -------------------------
def bernstein(n, i, t):
    from math import comb
    return comb(n, i) * (t**i) * ((1 - t)**(n - i))

def bezier_basis_matrix(m, n, us, vs):
    U, V = cp.meshgrid(us, vs, indexing='xy')
    U = U.flatten()
    V = V.flatten()
    nsamples = U.size
    nbasis = (m+1) * (n+1)
    A = cp.zeros((nsamples, nbasis), dtype=cp.float64)
    idx = 0
    for i in range(m+1):
        Bi = cp.array([bernstein(m, i, float(t)) for t in U])
        for j in range(n+1):
            Bj = cp.array([bernstein(n, j, float(t)) for t in V])
            A[:, idx] = Bi * Bj
            idx += 1
    return A

def fit_bezier_surface(img, degree_u=6, degree_v=6, downsample=8):
    h, w = img.shape
    ys = cp.linspace(0, h-1, max(2, h//downsample)).astype(int)
    xs = cp.linspace(0, w-1, max(2, w//downsample)).astype(int)
    sample_vals = img[cp.ix_(ys, xs)].astype(cp.float64)
    us = (xs.astype(cp.float64)) / (w - 1) if w > 1 else cp.array([0.0])
    vs = (ys.astype(cp.float64)) / (h - 1) if h > 1 else cp.array([0.0])
    A = bezier_basis_matrix(degree_u, degree_v, us, vs)
    s = sample_vals.flatten()
    p, *_ = cp.linalg.lstsq(A, s, rcond=None)
    us_full = cp.linspace(0.0, 1.0, w)
    vs_full = cp.linspace(0.0, 1.0, h)
    A_full = bezier_basis_matrix(degree_u, degree_v, us_full, vs_full)
    S_full = (A_full @ p).reshape((h, w))
    return S_full

# -------------------------
# BF parameter search (multi-scale grid)
# -------------------------
def select_best_bf_candidate(z, f_curr, sigma_space_grid, sigma_color_grid, eta, alpha, beta, radiation_center=None):
    best_J = 1e30
    best_b = None
    for ss in sigma_space_grid:
        for sc in sigma_color_grid:
            bf = bilateral_filter_img(z, ss, sc)
            residual = z - bf
            data_term = eta * cp.sum((z - f_curr - bf)**2)
            reg_img = alpha * R_image_hog(residual)
            reg_bias = beta * R_bias(bf, radiation_center=radiation_center)
            J = float(data_term) + reg_img + reg_bias
            if J < best_J:
                best_J = J
                best_b = bf
    return best_b, best_J

# -------------------------
# Progressive BFBSF algorithms
# -------------------------
def bfbsf_plus(z, n_iter=20, eta=0.2, alpha=0.6, beta=0.8, gamma=0.3,
               sigma_space_scales=[2,4,8], sigma_color_scales=[0.05,0.1,0.2],
               bezier_degree=(6,6), downsample=8, radiation_center=None, verbose=True):
    z = cp.array(z).astype(cp.float32)
    f_i = z.copy()
    biases = []
    if verbose:
        it = trange(n_iter, desc='BFBSF+')
    else:
        it = range(n_iter)
    for i in it:
        sigma_space_grid = []
        sigma_color_grid = []
        for s in sigma_space_scales:
            sigma_space_grid += [max(0.5, s * f) for f in [0.5,1.0,1.5]]
        for c in sigma_color_scales:
            sigma_color_grid += [max(0.01, c * f) for f in [0.5,1.0,2.0]]
        sigma_space_grid = sorted(set(sigma_space_grid))
        sigma_color_grid = sorted(set(sigma_color_grid))
        best_bf, bestJ = select_best_bf_candidate(z, f_i, sigma_space_grid, sigma_color_grid,
                                                  eta, alpha, beta, radiation_center=radiation_center)
        b_bezier = fit_bezier_surface(best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1],
                                      downsample=downsample)
        biases.append(b_bezier)
        f_i = f_i - gamma * b_bezier
        if verbose:
            it.set_postfix({'iter': i+1, 'J': float(bestJ)})
    return cp.asnumpy(f_i), [cp.asnumpy(b) for b in biases]

def bfbsf(z, n_iter=20, eta=0.2, alpha=0.6, beta=0.8, gamma=0.3,
          sigma_space_scales=[2,4,8], sigma_color_scales=[0.05,0.1,0.2],
          bezier_degree=(6,6), downsample=8, radiation_center=None, verbose=True):
    z = cp.array(z).astype(cp.float32)
    f_i = z.copy()
    biases = []
    sigma_space_grid = []
    sigma_color_grid = []
    for s in sigma_space_scales:
        sigma_space_grid += [max(0.5, s * f) for f in [0.5,1.0,1.5]]
    for c in sigma_color_scales:
        sigma_color_grid += [max(0.01, c * f) for f in [0.5,1.0,2.0]]
    sigma_space_grid = sorted(set(sigma_space_grid))
    sigma_color_grid = sorted(set(sigma_color_grid))
    best_bf, bestJ = select_best_bf_candidate(z, f_i, sigma_space_grid, sigma_color_grid,
                                              eta, alpha, beta, radiation_center=radiation_center)
    if verbose:
        print("Initial BF selection done, J=", bestJ)
    for i in trange(n_iter, desc='BFBSF' if verbose else None):
        b_bezier = fit_bezier_surface(best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1],
                                      downsample=downsample)
        biases.append(b_bezier)
        f_i = f_i - gamma * b_bezier
    return cp.asnumpy(f_i), [cp.asnumpy(b) for b in biases]

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io, img_as_float

    if torch.cuda.is_available():
        print(torch.version.cuda)
    else:
        print("CUDA illada Punda!!")
    img = img_as_float(io.imread("output_degraded.png", as_gray=True))
    img = np.clip(img, 0.0, 1.0)

    corrected_fast, biases_fast = bfbsf(img, n_iter=3, verbose=True)
    corrected_plus, biases_plus = bfbsf_plus(img, n_iter=3, verbose=True)

    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Original degraded")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("BFBSF (fast)")
    plt.imshow(np.clip(corrected_fast,0,1), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("BFBSF+ (full)")
    plt.imshow(np.clip(corrected_plus,0,1), cmap='gray')
    plt.axis('off')

    plt.show()