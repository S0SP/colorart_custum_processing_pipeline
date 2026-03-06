"""
Region Segmentation Module
----------------------------
Extracts connected-component regions from a quantized label map.
Each region is a contiguous blob of a single palette color.

Key outputs per region:
  - region_id      unique integer across ALL regions (1-indexed)
  - color_idx      which palette color this region belongs to
  - color_number   1-indexed color number shown in the app
  - mask           H×W boolean array
  - bbox           (x, y, w, h) bounding box
  - area           pixel count
  - contours       outer contour + holes (OpenCV format)
"""

import cv2
import numpy as np
import scipy.ndimage as ndi
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Region:
    region_id: int              # unique per region
    color_idx: int              # index into palette array
    color_number: int           # color number shown to user (1-indexed)
    color_hex: str              # e.g. "#ff6b9d"
    color_rgb: Tuple[int, int, int]
    area: int                   # pixel count
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    outer_contour: np.ndarray   # primary contour points (N,1,2) int32
    hole_contours: List[np.ndarray] = field(default_factory=list)
    label_pos: Optional[Tuple[float, float]] = None  # filled by labeler


def extract_regions(
    label_map: np.ndarray,
    palette: np.ndarray,
    min_area: int = 25,
    target_regions: Optional[int] = None,
) -> List[Region]:
    """
    Enhanced extraction: if target_regions is set, merges small blobs
    into neighbors until we hit the limit.
    """
    h, w = label_map.shape
    num_colors = palette.shape[0]

    # 1. Label every single contiguous blob in the entire image
    labeled_blobs = np.zeros_like(label_map, dtype=np.int32)
    blob_to_color = {}
    current_offset = 1
    
    for c_idx in range(num_colors):
        mask = (label_map == c_idx).astype(np.uint8)
        n, labels = cv2.connectedComponents(mask, connectivity=4)
        if n > 1:
            mask_bool = (mask > 0)
            labeled_blobs[mask_bool] = labels[mask_bool] + current_offset - 1
            for b_idx in range(1, n):
                blob_to_color[b_idx + current_offset - 1] = c_idx
            current_offset += (n - 1)

    num_blobs = current_offset - 1
    
    # 2. If target_regions is set, merge until we hit it
    if target_regions and num_blobs > target_regions:
        logger.info(f"Reducing {num_blobs} blobs to {target_regions}...")
        label_map = _smart_merge_blobs(
            label_map, labeled_blobs, blob_to_color, palette, target_regions, min_area
        )
        # We must re-extract regions from the updated label_map
        return extract_regions(label_map, palette, min_area=min_area, target_regions=None)

    # 3. Standard extraction loop (same as before but using the (possibly merged) label_map)
    regions: List[Region] = []
    region_id = 1

    for color_idx in range(num_colors):
        color_rgb = tuple(int(v) for v in palette[color_idx])
        color_hex = "#{:02x}{:02x}{:02x}".format(*color_rgb)
        color_number = color_idx + 1

        mask = (label_map == color_idx).astype(np.uint8)
        if not np.any(mask): continue

        n, labeled, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area: continue

            x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            bbox = (int(x), int(y), int(bw), int(bh))
            comp_mask = (labeled[y:y+bh, x:x+bw] == i).astype(np.uint8)
            padded = cv2.copyMakeBorder(comp_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            
            contours_raw, hierarchy = cv2.findContours(
                padded, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
            )
            
            if not contours_raw: continue
            outer_contour = None
            hole_contours = []
            h_arr = hierarchy[0] if hierarchy is not None else []
            for j, cnt in enumerate(contours_raw):
                if len(cnt) < 3: continue
                pts = cnt.astype(np.int32)
                pts[:, 0, 0] += (x - 1)
                pts[:, 0, 1] += (y - 1)
                simplified = cv2.approxPolyDP(pts, 0.4, closed=True)
                if h_arr[j][3] == -1:
                    if outer_contour is None or len(simplified) > len(outer_contour):
                        outer_contour = simplified
                else:
                    hole_contours.append(simplified)

            if outer_contour is not None:
                regions.append(Region(
                    region_id=region_id,
                    color_idx=color_idx,
                    color_number=color_number,
                    color_hex=color_hex,
                    color_rgb=color_rgb,
                    area=area,
                    bbox=bbox,
                    outer_contour=outer_contour,
                    hole_contours=hole_contours,
                ))
                region_id += 1

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _smart_merge_blobs(
    label_map: np.ndarray,
    labeled_blobs: np.ndarray,
    blob_to_color: dict,
    palette: np.ndarray,
    target: int,
    min_area: int,
) -> np.ndarray:
    """Find neighbors and merge smallest blobs until target count is reached."""
    import heapq
    from collections import defaultdict

    counts = np.bincount(labeled_blobs.ravel())
    # num_blobs = len(counts) - 1
    
    # Fast neighbor detection
    # Shift-based for massive speed up
    h, w = label_map.shape
    adj = defaultdict(set)
    for dy, dx in [(0,1), (1,0)]:
        a = labeled_blobs[:h-dy, :w-dx]
        b = labeled_blobs[dy:, dx:]
        mask = (a != b) & (a > 0) & (b > 0)
        for b1, b2 in np.unique(np.stack([a[mask], b[mask]], axis=1), axis=0):
            adj[int(b1)].add(int(b2))
            adj[int(b2)].add(int(b1))

    # Priority Queue for merging: (area, blob_id)
    pq = []
    active = set()
    for b_id in range(1, len(counts)):
        if counts[b_id] > 0:
            heapq.heappush(pq, (int(counts[b_id]), int(b_id)))
            active.add(b_id)

    num_active = len(active)
    
    # Merge until target
    # Note: This is an approximation. Real-time updates to PQ are slow, 
    # so we just merge the current smallest into their best neighbor.
    result_map = labeled_blobs.copy()
    
    while num_active > target and pq:
        area, b_id = heapq.heappop(pq)
        if b_id not in active: continue
        
        # Find best neighbor (closest color)
        neighbors = adj.get(b_id, set())
        if not neighbors:
            active.remove(b_id) # Island blob, nothing to do
            num_active -= 1
            continue
            
        c_i = blob_to_color[b_id]
        best_n = -1
        min_dist = float('inf')
        
        for n_id in neighbors:
            if n_id not in active: continue
            c_n = blob_to_color[n_id]
            dist = np.linalg.norm(palette[c_i].astype(float) - palette[c_n].astype(float))
            if dist < min_dist:
                min_dist = dist
                best_n = n_id
        
        if best_n != -1:
            # Merge b_id into best_n
            # In a real system we'd update adj and PQ properly, 
            # but for speed we just update the color reference.
            active.remove(b_id)
            result_map[result_map == b_id] = best_n
            num_active -= 1
        else:
            active.remove(b_id)
            num_active -= 1

    # Convert back to standard label_map colors
    final_label_map = np.zeros_like(label_map)
    for b_id in active:
        final_label_map[result_map == b_id] = blob_to_color[b_id]
        
    return final_label_map


def build_region_mask(region: Region, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [region.outer_contour], -1, 1, thickness=cv2.FILLED)
    for hole in region.hole_contours:
        cv2.drawContours(mask, [hole], -1, 0, thickness=cv2.FILLED)
    return mask.astype(bool)
