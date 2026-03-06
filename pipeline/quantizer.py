"""
Color Quantization Module
--------------------------
Reduces image to N dominant colors using KMeans.
Supports cartoon/illustration images (crisp color blocks)
and photos (smooth gradient regions).

Strategy:
  1. Optional bilateral filter to smooth noise while preserving edges
  2. KMeans on all pixels → N cluster centers
  3. Each pixel assigned to nearest cluster → quantized image
  4. Post-process: remove micro-regions by merging with dominant neighbor
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from PIL import Image
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def quantize_image(
    img_rgb: np.ndarray,
    num_colors: int = 12,
    is_illustration: bool = True,
    smooth_sigma: float = 1.0,
    min_region_fraction: float = 0.0002,
    target_regions: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantize image with 'Posterized Flow'.
    Adapts intensity based on target_regions to allow for extreme detail.
    """
    h, w = img_rgb.shape[:2]
    working = img_rgb.copy()

    # ── Step 1: Adaptive Flow Filter ──
    # If user wants lots of regions (>500), we must reduce smoothing
    if not is_illustration:
        t_reg = target_regions or 0
        
        if t_reg < 600:
            # Standard 'Posterized Flow' (Clean & Bold)
            working = cv2.bilateralFilter(working, d=9, sigmaColor=75, sigmaSpace=15)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            working = cv2.morphologyEx(working, cv2.MORPH_OPEN, kernel, iterations=1)
        elif t_reg < 1500:
            # Medium Detail
            working = cv2.bilateralFilter(working, d=5, sigmaColor=40, sigmaSpace=10)
        else:
            # Ultra Detail (Target 1500+): Almost no filtering
            working = cv2.bilateralFilter(working, d=3, sigmaColor=20, sigmaSpace=5)

        # Crisp re-sharpening
        working = cv2.addWeighted(working, 1.3, cv2.GaussianBlur(working, (0,0), 3), -0.3, 0)

    # ── Step 2: KMeans ──
    pixels = working.reshape(-1, 3).astype(np.float32)
    # Use larger sample for complex high-region requests
    sample_limit = 2000000 if (target_regions and target_regions > 1000) else 1000000
    sample_size = min(len(pixels), sample_size_limit := sample_limit)
    idx = np.random.choice(len(pixels), sample_size, replace=False)
    sample_pixels = pixels[idx]

    kmeans = MiniBatchKMeans(
        n_clusters=num_colors,
        random_state=42,
        batch_size=8192,
        max_iter=300,
        n_init=5,
    )
    kmeans.fit(sample_pixels)
    
    labels = kmeans.predict(pixels)
    palette = np.clip(kmeans.cluster_centers_, 0, 255).astype(np.uint8)

    # ── Step 3: Clean Label Map ──
    label_map = labels.reshape(h, w).astype(np.int32)
    
    # If target_regions is high, we MUST disable aggressive early merging
    effective_fraction = min_region_fraction
    if target_regions and target_regions > 800:
        effective_fraction = 0.00001 # Preserve almost everything for segmenter
        
    label_map = _merge_micro_regions(label_map, palette, num_colors, effective_fraction)

    quantized_img = palette[label_map]
    logger.info(f"Quantized (PRO FLOW) to {num_colors} colors, {w}x{h}")
    return quantized_img.astype(np.uint8), palette, label_map


def _merge_micro_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    num_colors: int,
    min_fraction: float,
) -> np.ndarray:
    """
    Cleaner merge for ribbon-like gradients.
    """
    import scipy.ndimage as ndi

    h, w = label_map.shape
    total_pixels = h * w
    # Allow merging ONLY the absolute noise if fraction is near zero
    min_pixels = max(1, int(total_pixels * min_fraction)) 

    # 1. Label components
    labeled_blobs = np.zeros_like(label_map, dtype=np.int32)
    current_offset = 0
    for color_idx in range(num_colors):
        mask = (label_map == color_idx).astype(np.uint8)
        n, labels = cv2.connectedComponents(mask, connectivity=8) 
        if n > 1:
            labeled_blobs[mask > 0] = labels[mask > 0] + current_offset
            current_offset += (n - 1)

    # 2. Sizes
    counts = np.bincount(labeled_blobs.ravel())
    tiny_blob_mask = (counts[labeled_blobs] < min_pixels) & (labeled_blobs > 0)
    
    if not np.any(tiny_blob_mask):
        return label_map

    # 3. Smooth fill
    dist, indices = ndi.distance_transform_edt(tiny_blob_mask, return_indices=True)
    result = label_map.copy()
    y_idx, x_idx = indices
    result[tiny_blob_mask] = label_map[y_idx[tiny_blob_mask], x_idx[tiny_blob_mask]]

    return result


def detect_illustration(img_rgb: np.ndarray) -> bool:
    """Fast illustration detector."""
    small = cv2.resize(img_rgb, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sample = small.reshape(-1, 3)
    unique_colors = len(np.unique(sample, axis=0))
    color_density = unique_colors / len(sample)
    
    # Relax illustration detection for cartoon-heavy images
    is_illus = laplacian_var > 300 and color_density < 0.25
    return bool(is_illus)
