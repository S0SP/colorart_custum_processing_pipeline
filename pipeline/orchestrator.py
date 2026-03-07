"""
Pipeline Orchestrator
----------------------
Runs the full image → SVG pipeline end-to-end.

Pipeline stages:
  1. Load + validate image
  2. Resize to processing resolution (keeps aspect ratio)
  3. Detect illustration vs photo
  4. Color quantization (KMeans)
  5. Region segmentation (connected components)
  6. Label placement (pole of inaccessibility)
  7. Adjacency graph construction
  8. SVG generation (colored + outline)
  9. Assemble response payload

All heavy work runs in <2s for a 1024px image on a modern CPU.
No GPU / ML model required — pure algorithmic pipeline.
"""

import time
import logging
import numpy as np
import cv2
from PIL import Image
import io
from typing import Optional

from pipeline.quantizer import quantize_image, detect_illustration
from pipeline.segmenter import extract_regions
from pipeline.labeler import compute_label_positions
from pipeline.adjacency import build_adjacency_graph
from pipeline.svg_builder import build_svgs, build_palette_legend_svg

logger = logging.getLogger(__name__)


# ── Tuneable defaults ──────────────────────────────────────
DEFAULT_PARAMS = {
    "num_colors": 32,            # Default to 32 for good detail coverage
    "max_dimension": 1024,
    "min_region_area": 25,       # Balance: small enough for eyes, big enough to hide noise
    "min_region_fraction": 0.0002, 
    "chaikin_iters": 2,          # Balanced smoothness
    "downsample_step": 1,        
    "min_font_size": 6,
    "max_font_size": 18,
    "base_font_size": 12,
}


def process_image(
    image_bytes: bytes,
    num_colors: Optional[int] = None,
    max_dimension: Optional[int] = None,
    min_region_area: Optional[int] = None,
    target_regions: Optional[int] = None,
) -> dict:
    """
    Full pipeline: raw image bytes → structured response dict.

    Returns dict with keys:
        width, height,
        svg_colored, svg_outline, svg_animated, svg_palette_legend,
        regions: [{region_id, color_number, color_hex, color_rgb,
                   area, label_x, label_y, label_font_size}],
        palette: [hex_string, ...],
        adjacency: {str(region_id): [neighbor_ids]}
        timing: {stage: seconds}
    """
    params = dict(DEFAULT_PARAMS)
    if num_colors is not None:
        params["num_colors"] = int(np.clip(num_colors, 4, 128))
    if max_dimension is not None:
        params["max_dimension"] = int(np.clip(max_dimension, 256, 2048))
    if min_region_area is not None:
        params["min_region_area"] = int(np.clip(min_region_area, 10, 5000))

    timing = {}
    t0 = time.perf_counter()

    # ── Stage 1: Load image ───────────────────────────────
    img_rgb = _load_image(image_bytes)
    timing["load"] = _elapsed(t0)

    # ── Stage 2: Resize ───────────────────────────────────
    img_rgb = _resize(img_rgb, params["max_dimension"])
    h, w = img_rgb.shape[:2]
    timing["resize"] = _elapsed(t0)

    # ── Stage 3: Detect image type ────────────────────────
    is_illus = detect_illustration(img_rgb)

    # ── Stage 4: Color quantization ───────────────────────
    _, palette, label_map = quantize_image(
        img_rgb,
        num_colors=params["num_colors"],
        is_illustration=is_illus,
        min_region_fraction=params["min_region_fraction"],
        target_regions=target_regions,
    )
    timing["quantize"] = _elapsed(t0)

    # ── Stage 5: Region segmentation ─────────────────────
    regions = extract_regions(
        label_map,
        palette,
        min_area=params["min_region_area"],
        target_regions=target_regions,
    )
    timing["segment"] = _elapsed(t0)

    if not regions:
        raise ValueError("No regions extracted — try reducing num_colors or min_region_area")

    # ── Stage 6: Label placement ──────────────────────────
    regions = compute_label_positions(
        regions,
        image_shape=(h, w),
        min_font_size=params["min_font_size"],
        max_font_size=params["max_font_size"],
        base_font_size=params["base_font_size"],
    )
    timing["labels"] = _elapsed(t0)

    # ── Stage 7: Adjacency graph ──────────────────────────
    adjacency = build_adjacency_graph(regions, image_shape=(h, w))
    timing["adjacency"] = _elapsed(t0)

    # ── Stage 8: SVG generation ───────────────────────────
    svg_colored, svg_outline, svg_animated, region_paths, mega_paths = build_svgs(
        regions,
        image_width=w,
        image_height=h,
        chaikin_iters=params["chaikin_iters"],
        downsample_step=params["downsample_step"],
    )
    palette_hex = ["#{:02x}{:02x}{:02x}".format(*c) for c in palette]
    svg_palette = build_palette_legend_svg(palette_hex)
    timing["svg"] = _elapsed(t0)

    # ── Stage 9: Elite Game Pre-computations ──────────────
    # 1. Quick Thumbnail (B64) for instant gallery loading
    import base64
    thumb = cv2.resize(palette[label_map], (256, int(256*h/w)), interpolation=cv2.INTER_AREA)
    _, thumb_buf = cv2.imencode(".jpg", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    thumbnail_b64 = base64.b64encode(thumb_buf).decode("utf-8")

    # 2. Area Analytics (for progress bars)
    total_area = h * w
    palette_stats = []
    for i in range(len(palette)):
        c_regions = [r for r in regions if r.color_idx == i]
        c_area = sum(r.area for r in c_regions)
        palette_stats.append({
            "color_idx": i,
            "hex": palette_hex[i],
            "area_fraction": round(c_area / total_area, 4),
            "region_count": len(c_regions),
        })

    # 3. Hint Priority: Sort regions by area (smallest = hardest)
    # Give high priority score to tiny hidden regions
    for r in regions:
        # Score between 0 (easy/large) and 100 (hard/tiny)
        r.hint_priority = max(0, min(100, int(100 * (1 - (r.area / (total_area * 0.01)))) ))

    # ── Stage 10: Assemble response ───────────────────────
    timing["total"] = _elapsed(t0)
    logger.info(f"Pipeline complete in {timing['total']:.3f}s — {len(regions)} regions")

    return {
        "width": w,
        "height": h,
        "thumbnail_b64": thumbnail_b64, # ELITE: For gallery
        "svg_colored": svg_colored,
        "svg_outline": svg_outline,
        "svg_animated": svg_animated,
        "mega_paths_by_color": {str(k): v for k, v in mega_paths.items()}, # ELITE: For 60FPS
        "regions": [
            {
                "region_id": r.region_id,
                "color_number": r.color_number,
                "color_idx": r.color_idx,
                "color_hex": r.color_hex,
                "path_data": region_paths.get(r.region_id, ""),
                "area": r.area,
                "hint_priority": getattr(r, "hint_priority", 50), # ELITE: For help system
                "bbox": {"x": r.bbox[0], "y": r.bbox[1], "w": r.bbox[2], "h": r.bbox[3]},
                "label_x": round(r.label_pos[0], 2) if r.label_pos else None,
                "label_y": round(r.label_pos[1], 2) if r.label_pos else None,
                "label_font_size": getattr(r, "label_font_size", 12),
            }
            for r in regions
        ],
        "palette": palette_hex,
        "palette_stats": palette_stats, # ELITE: For progress bars
        "adjacency": adjacency,
        "timing": {k: round(v, 4) for k, v in timing.items()},
        "meta": {
            "num_colors_requested": int(params["num_colors"]),
            "num_regions": len(regions),
            "is_illustration": bool(is_illus),
        },
    }


# ── Helpers ────────────────────────────────────────────────

def _load_image(image_bytes: bytes) -> np.ndarray:
    """Load image bytes → RGB numpy array."""
    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Cannot decode image: {e}")


def _resize(img_rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Resize so the longest side ≤ max_dim, preserving aspect ratio."""
    h, w = img_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img_rgb
    scale = max_dim / longest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _elapsed(t0: float) -> float:
    return time.perf_counter() - t0
