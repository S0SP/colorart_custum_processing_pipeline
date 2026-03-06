"""
Label Placement Module
-----------------------
Finds the optimal position to place a color number inside each region.

Algorithm: Pole of Inaccessibility
  The "pole of inaccessibility" is the point inside a polygon that is
  furthest from all edges — i.e., the center of the largest inscribed circle.
  This is computed using the Euclidean Distance Transform (EDT).

  For a connected component mask:
    1. Compute EDT of the mask (each pixel's distance to nearest boundary)
    2. The pixel with maximum EDT value is the pole of inaccessibility
    3. If the region is too narrow (max_dist < threshold), fall back to centroid

Additionally:
  - Font size is scaled to the inscribed circle radius so numbers fit
  - For very small regions, the number may be placed outside with a leader line
    (caller's responsibility to draw the line)
"""

import numpy as np
import scipy.ndimage as ndi
import cv2
from typing import Tuple, List, Optional
import logging

from pipeline.segmenter import Region, build_region_mask

logger = logging.getLogger(__name__)


def compute_label_positions(
    regions: List[Region],
    image_shape: Tuple[int, int],
    min_font_size: int = 8,
    max_font_size: int = 24,
    base_font_size: int = 14,
) -> List[Region]:
    """
    Compute labels using CROPPED masks for massive speedup.
    """
    for region in regions:
        # Efficiently generate ONLY the part of the mask we need
        bx, by, bw, bh = region.bbox
        
        # Draw on a local mini-mask
        mini_mask = np.zeros((bh, bw), dtype=np.uint8)
        
        # Shift contours to local space
        local_outer = region.outer_contour.copy()
        local_outer[:, 0, 0] -= bx
        local_outer[:, 0, 1] -= by
        
        cv2.drawContours(mini_mask, [local_outer], -1, 1, thickness=cv2.FILLED)
        for hole in region.hole_contours:
            local_hole = hole.copy()
            local_hole[:, 0, 0] -= bx
            local_hole[:, 0, 1] -= by
            cv2.drawContours(mini_mask, [local_hole], -1, 0, thickness=cv2.FILLED)

        lx_local, ly_local, font_size = _pole_of_inaccessibility(
            mini_mask, min_font_size, max_font_size, base_font_size
        )
        
        # Translate back to global
        region.label_pos = (float(lx_local + bx), float(ly_local + by))
        region.label_font_size = font_size

    return regions


def _pole_of_inaccessibility(
    mask: np.ndarray,
    min_font: int,
    max_font: int,
    base_font: int,
) -> Tuple[float, float, int]:
    """
    Find the visual center of a binary mask using the EDT approach.

    Returns:
        (x, y, font_size) — float coordinates and recommended font size
    """
    mask_u8 = mask.astype(np.uint8)

    # Euclidean distance transform: each pixel = distance to nearest 0
    edt = ndi.distance_transform_edt(mask_u8)

    # Inscribed circle radius = max EDT value
    max_dist = float(edt.max())

    if max_dist < 3:
        # Degenerate region — use centroid
        y_c, x_c = np.array(np.where(mask_u8)).mean(axis=1)
        return float(x_c), float(y_c), min_font

    # Find all pixels at/near the maximum distance
    peak_threshold = max_dist * 0.92
    candidates = np.argwhere(edt >= peak_threshold)  # (N, 2) in [row, col]

    if len(candidates) == 0:
        y_c, x_c = np.array(np.where(mask_u8)).mean(axis=1)
        return float(x_c), float(y_c), min_font

    # Among candidates, choose the one closest to the overall centroid
    # (breaks ties, avoids picking a point in a weird appendage)
    rows = np.where(mask_u8)[0]
    cols = np.where(mask_u8)[1]
    centroid = np.array([rows.mean(), cols.mean()])

    dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
    best_idx = np.argmin(dists_to_centroid)
    best_yx = candidates[best_idx]

    x = float(best_yx[1])
    y = float(best_yx[0])

    # Scale font size to inscribed circle
    # A circle of radius R comfortably fits a 2-digit number at ~R*0.9 px
    font_size = int(np.clip(max_dist * 0.9, min_font, max_font))

    return x, y, font_size


def find_label_outside_region(
    region: Region,
    image_shape: Tuple[int, int],
    margin: int = 10,
) -> Optional[Tuple[float, float]]:
    """
    For very tiny regions, suggest a label position just outside the region
    (e.g., above or to the right of the bounding box).
    Returns None if there's sufficient space inside.
    """
    if region.label_font_size >= 10:
        return None  # fits inside

    bx, by, bw, bh = region.bbox
    ih, iw = image_shape

    # Try above first
    lx = bx + bw / 2
    ly = by - margin
    if ly > 0:
        return (lx, ly)

    # Try right
    lx = bx + bw + margin
    ly = by + bh / 2
    if lx < iw:
        return (lx, ly)

    # Fall back to original inside position
    return None
