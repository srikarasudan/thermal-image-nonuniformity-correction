"""
bfbsf_cuda.py

CUDA implementation of BFBSF+ and BFBSF using PyTorch
Converted from numpy/skimage implementation to leverage GPU acceleration.

Key changes:
- All operations moved to PyTorch tensors with CUDA support
- Custom bilateral filter implementation using PyTorch
- GPU-accelerated HOG and gradient computations
- Bézier surface fitting with batched matrix operations
- Proper tensor shape and dtype handling throughout

Author: Converted to CUDA/PyTorch
"""

from typing import Tuple, List, Optional, Union
import torch
import torch.nn.functional as F
import math
import numpy as np
from tqdm import trange

# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------
# Utility functions (CUDA)
# -------------------------

def ensure_tensor(img: Union[np.ndarray, torch.Tensor], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to CUDA tensor with proper dtype"""
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    
    img = img.to(device=device, dtype=dtype)
    
    # Normalize to [0,1] if needed
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    elif img.min() < 0.0 or img.max() > 1.0:
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min + 1e-12)
    
    return img

def gaussian_kernel_2d(size: int, sigma: float) -> torch.Tensor:
    """Create 2D Gaussian kernel on CUDA"""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.outer(g)  # 2D kernel

def bilateral_filter_cuda(img: torch.Tensor, sigma_space: float, sigma_color: float, 
                         kernel_size: Optional[int] = None) -> torch.Tensor:
    """
    CUDA implementation of bilateral filtering
    Args:
        img: 2D tensor [H, W] in range [0,1]
        sigma_space: spatial sigma in pixels
        sigma_color: color sigma in intensity units [0,1]
        kernel_size: spatial kernel size (auto if None)
    """
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma_space) + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    H, W = img.shape
    half_k = kernel_size // 2
    
    # Pad image
    img_pad = F.pad(img.unsqueeze(0), (half_k, half_k, half_k, half_k), mode='reflect').squeeze(0)
    
    # Create spatial weights
    spatial_kernel = gaussian_kernel_2d(kernel_size, sigma_space)  # [kernel_size, kernel_size]
    
    # Output tensor
    result = torch.zeros_like(img)
    
    # Sliding window approach (vectorized)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Extract shifted patch
            patch = img_pad[i:i+H, j:j+W]  # [H, W]
            
            # Compute color weights
            color_diff = torch.abs(patch - img)  # [H, W]
            color_weight = torch.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))
            
            # Combined weight
            spatial_w = spatial_kernel[i, j]
            combined_weight = spatial_w * color_weight  # [H, W]
            
            # Accumulate weighted values
            result += combined_weight * patch
    
    # Normalize
    # Compute normalization weights
    norm_weights = torch.zeros_like(img)
    for i in range(kernel_size):
        for j in range(kernel_size):
            patch = img_pad[i:i+H, j:j+W]
            color_diff = torch.abs(patch - img)
            color_weight = torch.exp(-(color_diff ** 2) / (2 * sigma_color ** 2))
            spatial_w = spatial_kernel[i, j]
            norm_weights += spatial_w * color_weight
    
    result = result / (norm_weights + 1e-12)
    return result

def gradient_and_orientation_cuda(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute gradient magnitude and orientation using CUDA
    Args:
        img: 2D tensor [H, W]
    Returns:
        magnitude: [H, W] gradient magnitudes
        orientation: [H, W] orientations in [0, 2π)
    """
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Add batch and channel dimensions for conv2d
    img_4d = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # Compute gradients
    gx = F.conv2d(img_4d, sobel_x, padding=1).squeeze()  # [H, W]
    gy = F.conv2d(img_4d, sobel_y, padding=1).squeeze()  # [H, W]
    
    # Magnitude and orientation
    mag = torch.sqrt(gx**2 + gy**2)
    ori = torch.atan2(gy, gx)  # [-π, π]
    ori = torch.fmod(ori + 2*math.pi, 2*math.pi)  # [0, 2π)
    
    return mag, ori

# -------------------------
# Regularizers (CUDA)
# -------------------------

def R_bias_cuda(bias: torch.Tensor, radiation_center: Optional[tuple] = None) -> torch.Tensor:
    """
    CUDA implementation of bias-field regularizer
    """
    H, W = bias.shape
    mag, ori = gradient_and_orientation_cuda(bias)
    
    # L1 gradient norm
    l1_grad = torch.sum(torch.abs(mag))
    
    # Radiation center
    if radiation_center is None:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    else:
        cx, cy = radiation_center
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Vector to center
    vec_x = cx - x_coords
    vec_y = cy - y_coords
    theta_center = torch.fmod(torch.atan2(vec_y, vec_x) + 2*math.pi, 2*math.pi)
    
    # Angular alignment
    angle_diff = torch.abs(torch.angle(torch.exp(1j * (ori - theta_center))))
    aligned = (angle_diff <= (math.pi / 4.0)).float()
    orientation_score = torch.sum(aligned * mag)
    
    reg = l1_grad - orientation_score
    return reg

def compute_hog_cuda(patch: torch.Tensor, n_bins: int = 9) -> torch.Tensor:
    """
    Compute HOG features for a patch using CUDA
    """
    if patch.numel() == 0:
        return torch.zeros(n_bins, dtype=torch.float32, device=device)
    
    # Compute gradients
    mag, ori = gradient_and_orientation_cuda(patch)
    
    # Create histogram
    bin_edges = torch.linspace(0, 2*math.pi, n_bins + 1, device=device)
    hist = torch.zeros(n_bins, dtype=torch.float32, device=device)
    
    for i in range(n_bins):
        mask = (ori >= bin_edges[i]) & (ori < bin_edges[i + 1])
        hist[i] = torch.sum(mag[mask])
    
    # Normalize
    hist = hist / (torch.sum(hist) + 1e-12)
    return hist

def R_image_hog_cuda(img: torch.Tensor) -> torch.Tensor:
    """
    CUDA implementation of HOG-based image regularizer
    """
    H, W = img.shape
    cx, cy = W // 2, H // 2
    
    # Define 4 quadrants
    blocks = [
        (0, cy, 0, cx),      # top-left
        (0, cy, cx, W),      # top-right
        (cy, H, cx, W),      # bottom-right
        (cy, H, 0, cx),      # bottom-left
    ]
    
    hogs = []
    for (y0, y1, x0, x1) in blocks:
        patch = img[y0:y1, x0:x1]
        hog_vec = compute_hog_cuda(patch)
        hogs.append(hog_vec)
    
    # Stack HOG vectors
    H_matrix = torch.stack(hogs)  # [4, n_bins]
    mean_vec = torch.mean(H_matrix, dim=0, keepdim=True)  # [1, n_bins]
    
    # Compute deviations
    diffs = H_matrix - mean_vec  # [4, n_bins]
    total = torch.sum(diffs ** 2)
    
    return total

# -------------------------
# Bézier surface fitting (CUDA)
# -------------------------

def bernstein_cuda(n: int, i: int, t: torch.Tensor) -> torch.Tensor:
    """CUDA Bernstein polynomial computation"""
    from math import comb
    coeff = comb(n, i)
    return coeff * (t ** i) * ((1 - t) ** (n - i))

def bezier_basis_matrix_cuda(m: int, n: int, us: torch.Tensor, vs: torch.Tensor) -> torch.Tensor:
    """
    Create Bézier basis matrix on CUDA
    """
    U, V = torch.meshgrid(us, vs, indexing='xy')
    U = U.flatten()  # [nsamples]
    V = V.flatten()  # [nsamples]
    
    nsamples = U.shape[0]
    nbasis = (m + 1) * (n + 1)
    
    A = torch.zeros((nsamples, nbasis), dtype=torch.float64, device=device)
    
    idx = 0
    for i in range(m + 1):
        Bi = bernstein_cuda(m, i, U.double())  # [nsamples]
        for j in range(n + 1):
            Bj = bernstein_cuda(n, j, V.double())  # [nsamples]
            A[:, idx] = Bi * Bj
            idx += 1
    
    return A

def fit_bezier_surface_cuda(img: torch.Tensor, degree_u: int = 6, degree_v: int = 6, 
                           downsample: int = 8, ridge: float = 1e-6) -> torch.Tensor:
    """
    CUDA implementation of Bézier surface fitting
    """
    H, W = img.shape
    nbasis = (degree_u + 1) * (degree_v + 1)
    
    # Determine sampling grid
    samp_h = max(2, H // downsample)
    samp_w = max(2, W // downsample)
    
    if samp_h * samp_w < nbasis:
        side = int(math.ceil(math.sqrt(nbasis)))
        samp_h = min(H, side)
        samp_w = min(W, side)
    
    # Sample points
    ys = torch.linspace(0, H-1, samp_h, dtype=torch.long, device=device)
    xs = torch.linspace(0, W-1, samp_w, dtype=torch.long, device=device)
    
    # Extract sample values
    Y_grid, X_grid = torch.meshgrid(ys, xs, indexing='ij')
    sample_vals = img[Y_grid, X_grid].double().flatten()  # [nsamples]
    
    # Parameter space coordinates
    us = xs.double() / (W - 1) if W > 1 else torch.tensor([0.0], device=device, dtype=torch.float64)
    vs = ys.double() / (H - 1) if H > 1 else torch.tensor([0.0], device=device, dtype=torch.float64)
    
    # Build basis matrix
    A = bezier_basis_matrix_cuda(degree_u, degree_v, us, vs)  # [nsamples, nbasis]
    
    # Regularized least squares
    ATA = A.T @ A  # [nbasis, nbasis]
    ATA += ridge * torch.eye(ATA.shape[0], dtype=torch.float64, device=device)
    rhs = A.T @ sample_vals  # [nbasis]
    
    # Solve system
    try:
        p = torch.linalg.solve(ATA, rhs)  # [nbasis]
    except:
        # Fallback to pseudoinverse if singular
        p = torch.linalg.pinv(ATA) @ rhs
    
    # Evaluate at full resolution
    us_full = torch.linspace(0.0, 1.0, W, dtype=torch.float64, device=device)
    vs_full = torch.linspace(0.0, 1.0, H, dtype=torch.float64, device=device)
    
    A_full = bezier_basis_matrix_cuda(degree_u, degree_v, us_full, vs_full)  # [H*W, nbasis]
    S_full = (A_full @ p).view(H, W)  # [H, W]
    
    return S_full.float()

# -------------------------
# BF parameter search (CUDA)
# -------------------------

def select_best_bf_candidate_cuda(z: torch.Tensor, f_curr: torch.Tensor,
                                 sigma_space_grid: List[float], sigma_color_grid: List[float],
                                 eta: float, alpha: float, beta: float, 
                                 radiation_center: Optional[tuple] = None) -> Tuple[torch.Tensor, float]:
    """
    CUDA implementation of bilateral filter parameter search
    """
    best_J = float('inf')
    best_b = None
    
    for ss in sigma_space_grid:
        for sc in sigma_color_grid:
            # Apply bilateral filter
            bf = bilateral_filter_cuda(z, sigma_space=ss, sigma_color=sc)
            
            # Compute objective terms
            residual = z - bf
            data_term = eta * torch.sum((z - f_curr - bf) ** 2)
            reg_img = alpha * R_image_hog_cuda(residual)
            reg_bias = beta * R_bias_cuda(bf, radiation_center=radiation_center)
            
            J = data_term + reg_img + reg_bias
            J_val = float(J.item())
            
            if J_val < best_J:
                best_J = J_val
                best_b = bf.clone()
    
    return best_b, best_J

# -------------------------
# Progressive BFBSF algorithms (CUDA)
# -------------------------

def _build_sigma_grids(space_scales: List[float], color_scales: List[float]) -> Tuple[List[float], List[float]]:
    """Build parameter grids for BF search"""
    sigma_space_grid = sorted(set([max(0.5, s * factor) for s in space_scales for factor in (0.5, 1.0, 1.5)]))
    sigma_color_grid = sorted(set([max(1e-4, c * factor) for c in color_scales for factor in (0.5, 1.0, 2.0)]))
    return sigma_space_grid, sigma_color_grid

def bfbsf_plus_cuda(z: Union[np.ndarray, torch.Tensor], n_iter: int = 20, 
                   eta: float = 0.2, alpha: float = 0.6, beta: float = 0.8, gamma: float = 0.3,
                   sigma_space_scales: List[float] = [2, 4, 8], 
                   sigma_color_scales: List[float] = [0.02, 0.05, 0.12],
                   bezier_degree: Tuple[int, int] = (6, 6), downsample: int = 8, 
                   radiation_center: Optional[tuple] = None, verbose: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    CUDA implementation of BFBSF+ (re-search BF params each iteration)
    """
    z = ensure_tensor(z)
    f_i = z.clone()
    biases = []
    
    sigma_space_grid, sigma_color_grid = _build_sigma_grids(sigma_space_scales, sigma_color_scales)
    
    it_range = trange(n_iter, desc='BFBSF+ (CUDA)') if verbose else range(n_iter)
    
    for i in it_range:
        # Find best bilateral filter parameters
        best_bf, best_J = select_best_bf_candidate_cuda(
            z, f_i, sigma_space_grid, sigma_color_grid, eta, alpha, beta, radiation_center
        )
        
        # Fit Bézier surface
        b_bezier = fit_bezier_surface_cuda(
            best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1], 
            downsample=downsample, ridge=1e-6
        )
        
        biases.append(b_bezier.clone())
        f_i = f_i - gamma * b_bezier
        
        if verbose:
            it_range.set_postfix({'iter': i+1, 'J': f'{best_J:.6f}'})
    
    return f_i, biases

def bfbsf_cuda(z: Union[np.ndarray, torch.Tensor], n_iter: int = 20,
              eta: float = 0.2, alpha: float = 0.6, beta: float = 0.8, gamma: float = 0.3,
              sigma_space_scales: List[float] = [2, 4, 8], 
              sigma_color_scales: List[float] = [0.02, 0.05, 0.12],
              bezier_degree: Tuple[int, int] = (6, 6), downsample: int = 8, 
              radiation_center: Optional[tuple] = None, verbose: bool = True) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    CUDA implementation of BFBSF (search BF params once, reuse)
    """
    z = ensure_tensor(z)
    f_i = z.clone()
    biases = []
    
    sigma_space_grid, sigma_color_grid = _build_sigma_grids(sigma_space_scales, sigma_color_scales)
    
    # Initial BF parameter search
    best_bf, best_J = select_best_bf_candidate_cuda(
        z, f_i, sigma_space_grid, sigma_color_grid, eta, alpha, beta, radiation_center
    )
    
    if verbose:
        print(f"Initial BF selection done, J={best_J:.6f}")
    
    # Iterative bias correction
    for i in trange(n_iter, desc='BFBSF (CUDA)' if verbose else None):
        # Fit Bézier surface to the fixed bilateral filter result
        b_bezier = fit_bezier_surface_cuda(
            best_bf, degree_u=bezier_degree[0], degree_v=bezier_degree[1], 
            downsample=downsample, ridge=1e-6
        )
        
        biases.append(b_bezier.clone())
        f_i = f_i - gamma * b_bezier
    
    return f_i, biases

# -------------------------
# Utility functions for numpy compatibility
# -------------------------

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert CUDA tensor back to numpy array"""
    return tensor.detach().cpu().numpy()

def to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensor to CPU"""
    return tensor.detach().cpu()

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Your degraded image (can be numpy array or torch tensor)
    degraded_image = plt.imread('degraded_image.png')  # Example path
    # Run BFBSF (faster, single BF parameter search)
    corrected_fast, biases_fast = bfbsf_cuda(
        degraded_image, 
        n_iter=3, 
        verbose=True
    )

    # Run BFBSF+ (full, re-search parameters each iteration)
    corrected_plus, biases_plus = bfbsf_plus_cuda(
        degraded_image, 
        n_iter=2, 
        verbose=True
    )

    
    # Convert to numpy for visualization
    I_np = to_numpy(degraded_image)
    corrected_fast_np = to_numpy(torch.clamp(corrected_fast, 0, 1))
    corrected_plus_np = to_numpy(torch.clamp(corrected_plus, 0, 1))
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Degraded")
    plt.imshow(I_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("BFBSF (CUDA)")
    plt.imshow(corrected_fast_np, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("BFBSF+ (CUDA)")
    plt.imshow(corrected_plus_np, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nProcessing complete!")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6:.1f} MB")