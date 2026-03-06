"""
Vectorizer Module
------------------
Converts raw pixel contours into smooth SVG cubic Bézier paths.

Algorithm:
  1. Chaikin corner-cutting (3 iterations) → smooth polyline
  2. Catmull-Rom → Cubic Bézier conversion for SVG C commands
  3. Hole contours handled with even-odd fill rule

Why Catmull-Rom → Bezier?
  Catmull-Rom splines pass through every control point (unlike pure Bezier)
  and produce the most natural-looking smooth curves — matching the
  illustration style visible in the sample images.

Key tuning:
  - chaikin_iters: 3 gives good smoothing; use 2 for sharper corners
  - downsample_step: reduce contour point density before smoothing
    (too many points = overfitting, too few = angular)
  - tension: Catmull-Rom tension; 0.5 = standard centripetal
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def contour_to_svg_path(
    outer_contour: np.ndarray,
    hole_contours: List[np.ndarray],
    chaikin_iters: int = 3,
    downsample_step: int = 2,
) -> str:
    """
    Convert an outer contour + optional holes into a single compound SVG path string.
    Uses Catmull-Rom splines for silky smooth curves.

    Returns:
        SVG path data string (e.g. "M x,y C ... Z M x,y C ... Z")
        Uses even-odd fill rule so holes appear transparent.
    """
    parts = []

    # Outer boundary
    outer_pts = _prepare_points(outer_contour, downsample_step)
    if outer_pts is not None and len(outer_pts) >= 3:
        smoothed = _chaikin(outer_pts, chaikin_iters)
        parts.append(_points_to_catmull_bezier(smoothed, closed=True))

    # Holes (wound in reverse so even-odd rule punches them out)
    for hole in hole_contours:
        pts = _prepare_points(hole, downsample_step)
        if pts is not None and len(pts) >= 3:
            pts_rev = pts[::-1]  # reverse winding
            smoothed = _chaikin(pts_rev, chaikin_iters)
            parts.append(_points_to_catmull_bezier(smoothed, closed=True))

    return " ".join(parts) if parts else ""


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _prepare_points(
    contour: np.ndarray,
    step: int = 2,
) -> Optional[np.ndarray]:
    """
    Reshape OpenCV contour (N,1,2) → (M,2) float array,
    downsampled by `step` to reduce over-fitting.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return None
    # Downsample while keeping first/last
    if len(pts) > 8 and step > 1:
        idx = list(range(0, len(pts), step))
        if idx[-1] != len(pts) - 1:
            idx.append(len(pts) - 1)
        pts = pts[idx]
    return pts


def _chaikin(pts: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Chaikin's corner-cutting algorithm.
    Each iteration replaces every segment AB with two points: 0.75A+0.25B and 0.25A+0.75B.
    Result is a quadratic B-spline approximation — very smooth.
    """
    pts = pts.copy()
    for _ in range(iterations):
        n = len(pts)
        new_pts = np.empty((n * 2, 2), dtype=np.float64)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            new_pts[i * 2]     = 0.75 * a + 0.25 * b
            new_pts[i * 2 + 1] = 0.25 * a + 0.75 * b
        pts = new_pts
    return pts


def _points_to_catmull_bezier(pts: np.ndarray, closed: bool = True) -> str:
    """
    Convert a polyline of points to an SVG path using cubic Bézier curves
    derived from the Catmull-Rom formulation.

    Each segment P[i] → P[i+1] uses control points:
        CP1 = P[i]   + (P[i+1] - P[i-1]) / 6
        CP2 = P[i+1] - (P[i+2] - P[i])   / 6

    This ensures C1 continuity (tangent continuity) at every knot.
    """
    n = len(pts)
    if n < 2:
        return ""

    # fmt helper
    def f(v: float) -> str:
        return f"{v:.3f}"

    segments = []
    start = pts[0]
    segments.append(f"M {f(start[0])},{f(start[1])}")

    num_segs = n if closed else n - 1

    for i in range(num_segs):
        p0 = pts[(i - 1) % n]
        p1 = pts[i % n]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]

        # Catmull-Rom → Cubic Bezier control points
        cp1 = p1 + (p2 - p0) / 6.0
        cp2 = p2 - (p3 - p1) / 6.0
        end = p2

        segments.append(
            f"C {f(cp1[0])},{f(cp1[1])} {f(cp2[0])},{f(cp2[1])} {f(end[0])},{f(end[1])}"
        )

    if closed:
        segments.append("Z")

    return " ".join(segments)


def estimate_path_complexity(contour: np.ndarray) -> int:
    """Returns approximate number of bezier segments that would be generated."""
    pts = _prepare_points(contour, step=2)
    if pts is None:
        return 0
    # Chaikin triples points each iteration × 3 iters ÷ by expected reuse
    return len(pts) * (2 ** 3)
