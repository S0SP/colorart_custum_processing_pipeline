"""
Adjacency Graph Module
-----------------------
Builds a region adjacency graph (RAG) where nodes are region_ids
and edges indicate that two regions share a border.

Algorithm:
  For each region mask, dilate by 1 pixel. Any overlapping pixel
  between a dilated region and another (non-dilated) region means
  the two are adjacent.

  Optimized: instead of O(N²) pairwise comparison, we use a
  border-label scan — scan all pixels in dilation boundaries
  and record which two region IDs meet there.

Output:
  {
    "1": [2, 5, 7],   # region 1 is adjacent to 2, 5, 7
    "2": [1, 3],
    ...
  }
"""

import numpy as np
import cv2
import scipy.ndimage as ndi
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import logging

from pipeline.segmenter import Region

logger = logging.getLogger(__name__)


def build_adjacency_graph(
    regions: List[Region],
    image_shape: Tuple[int, int],
    dilation_px: int = 1,
) -> Dict[str, List[int]]:
    """
    Build region adjacency graph using a fast shift-based pixel lookup.
    """
    h, w = image_shape
    id_map = np.zeros((h, w), dtype=np.int32)

    # 1. Faster ID mapping without re-drawing contours
    # We can use the existing Region objects but wait - 
    # the labels are already in the quantized label_map.
    # Actually, we should use the same component logic used in extractor.
    # But wait, we already have a list of Regions.
    # We can just draw them once to create a final ID map.
    for r in regions:
        cv2.drawContours(id_map, [r.outer_contour], -1, r.region_id, thickness=cv2.FILLED)
        for hole in r.hole_contours:
            cv2.drawContours(id_map, [hole], -1, 0, thickness=cv2.FILLED)

    # 2. Fast Adjacency via Shifting
    # Shift the id_map in 4 (or 8) directions and find where colors meet.
    adjacency: Dict[int, Set[int]] = defaultdict(set)

    # Horizontal and Vertical shifts
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in offsets:
        # shifted slice
        if dy == 1 and dx == 0:
            a = id_map[:-1, :]
            b = id_map[1:, :]
        elif dy == 0 and dx == 1:
            a = id_map[:, :-1]
            b = id_map[:, 1:]
        elif dy == 1 and dx == 1:
            a = id_map[:-1, :-1]
            b = id_map[1:, 1:]
        elif dy == 1 and dx == -1:
            a = id_map[:-1, 1:]
            b = id_map[1:, :-1]
        
        mask = (a != b) & (a > 0) & (b > 0)
        # Unique pairs
        unique_pairs = np.unique(np.stack([a[mask], b[mask]], axis=1), axis=0)
        for rid1, rid2 in unique_pairs:
            adjacency[rid1].add(int(rid2))
            adjacency[rid2].add(int(rid1))

    # 3. Format result
    result: Dict[str, List[int]] = {}
    for r in regions:
        rid = r.region_id
        result[str(rid)] = sorted(list(adjacency.get(rid, set())))

    logger.info(f"Adjacency graph: {len(result)} nodes")
    return result
