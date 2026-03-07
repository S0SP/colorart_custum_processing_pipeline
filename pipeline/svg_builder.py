"""
SVG Builder Module
-------------------
Assembles the final SVG documents from processed regions.

Two SVG outputs:
  1. svg_colored  — filled with correct colors, black outlines (reference image)
  2. svg_outline  — white fill, black outlines + centered numbers (the coloring canvas)

SVG structure:
  <svg viewBox="0 0 W H" xmlns="...">
    <defs>
      <!-- clipPath per region for crisp boundaries -->
    </defs>
    <g id="regions">
      <path id="region-{id}" d="..." fill="{color}" stroke="#1a1a1a"
            stroke-width="1.2" stroke-linejoin="round"
            fill-rule="evenodd"
            data-region-id="{id}"
            data-color-number="{number}"
            data-color="{hex}" />
      ...
    </g>
    <g id="labels">  <!-- outline SVG only -->
      <text x="..." y="..." ...>{number}</text>
      ...
    </g>
  </svg>

Design choices:
  - stroke-width: 1.2px — crisp outlines that look clean on any screen density
  - fill-rule: evenodd — handles holes correctly
  - stroke-linejoin: round — smooth outline corners
  - All coordinates are float with 3 decimal places for compactness
  - Label text is centered (text-anchor: middle, dominant-baseline: central)
"""

from typing import List, Dict, Tuple, Optional
import logging

from pipeline.segmenter import Region
from pipeline.vectorizer import contour_to_svg_path

logger = logging.getLogger(__name__)

# ── SVG constants ──────────────────────────────────────────
STROKE_COLOR = "#1a1a1a"
STROKE_WIDTH = "1.2"
OUTLINE_FILL = "#ffffff"
LABEL_FONT_FAMILY = "Arial, Helvetica, sans-serif"
LABEL_FONT_WEIGHT = "bold"
LABEL_COLOR = "#222222"
BACKGROUND_COLOR = "#ffffff"


def build_svgs(
    regions: List[Region],
    image_width: int,
    image_height: int,
    chaikin_iters: int = 3,
    downsample_step: int = 2,
) -> Tuple[str, str, str, Dict[int, str], Dict[int, str]]:
    """
    Build colored, outline and animated SVG strings.

    Returns:
        (svg_colored, svg_outline, svg_animated, region_paths, paths_by_color) 
    """
    # 1. Pre-compute all individual region paths
    region_paths: Dict[int, str] = {}
    # 2. MEGA-PATH: Group paths by color_idx for 60FPS frontend performance
    from collections import defaultdict
    color_groups = defaultdict(list)

    for region in regions:
        path_d = contour_to_svg_path(
            region.outer_contour,
            region.hole_contours,
            chaikin_iters=chaikin_iters,
            downsample_step=downsample_step,
        )
        if path_d:
            region_paths[region.region_id] = path_d
            color_groups[region.color_idx].append(path_d)

    # Join paths with a space - this is a valid SVG "multi-path"
    paths_by_color = {
        c_idx: " ".join(d_list) for c_idx, d_list in color_groups.items()
    }

    svg_colored = _build_colored_svg(regions, region_paths, image_width, image_height)
    svg_outline = _build_outline_svg(regions, region_paths, image_width, image_height)
    svg_animated = _build_animated_svg(regions, region_paths, image_width, image_height)

    logger.info(f"Built SVGs: {image_width}x{image_height}, {len(regions)} regions")
    return svg_colored, svg_outline, svg_animated, region_paths, paths_by_color


def _svg_header(w: int, h: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" '
        f'shape-rendering="geometricPrecision" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
    )


def _background_rect(w: int, h: int, fill: str) -> str:
    return f'  <rect width="{w}" height="{h}" fill="{fill}"/>\n'


def _build_colored_svg(
    regions: List[Region],
    region_paths: Dict[int, str],
    w: int,
    h: int,
) -> str:
    # Use the color of the first region (usually the largest background) 
    # as the SVG's background color to hide gaps.
    bg_color = regions[0].color_hex if regions else BACKGROUND_COLOR
    
    parts = [_svg_header(w, h)]
    parts.append(_background_rect(w, h, bg_color))
    parts.append('  <g id="regions">\n')

    for region in regions:
        path_d = region_paths.get(region.region_id)
        if not path_d:
            continue

        parts.append(
            f'    <path\n'
            f'      id="region-{region.region_id}"\n'
            f'      d="{path_d}"\n'
            f'      fill="{region.color_hex}"\n'
            f'      stroke="{STROKE_COLOR}"\n'
            f'      stroke-width="{STROKE_WIDTH}"\n'
            f'      stroke-linejoin="round"\n'
            f'      fill-rule="evenodd"\n'
            f'      data-region-id="{region.region_id}"\n'
            f'      data-color-number="{region.color_number}"\n'
            f'      data-color="{region.color_hex}"\n'
            f'    />\n'
        )

    parts.append('  </g>\n')
    parts.append('</svg>')
    return "".join(parts)


def _build_outline_svg(
    regions: List[Region],
    region_paths: Dict[int, str],
    w: int,
    h: int,
) -> str:
    parts = [_svg_header(w, h)]
    parts.append(_background_rect(w, h, OUTLINE_FILL))
    parts.append('  <g id="regions">\n')

    for region in regions:
        path_d = region_paths.get(region.region_id)
        if not path_d:
            continue

        parts.append(
            f'    <path\n'
            f'      id="region-{region.region_id}"\n'
            f'      d="{path_d}"\n'
            f'      fill="{OUTLINE_FILL}"\n'
            f'      stroke="{STROKE_COLOR}"\n'
            f'      stroke-width="{STROKE_WIDTH}"\n'
            f'      stroke-linejoin="round"\n'
            f'      fill-rule="evenodd"\n'
            f'      data-region-id="{region.region_id}"\n'
            f'      data-color-number="{region.color_number}"\n'
            f'      data-color="{region.color_hex}"\n'
            f'    />\n'
        )

    parts.append('  </g>\n')

    # ── Number labels ──────────────────────────────────────
    parts.append('  <g id="labels">\n')

    for region in regions:
        if region.label_pos is None:
            continue

        lx, ly = region.label_pos
        fs = getattr(region, 'label_font_size', 12)
        number = str(region.color_number)

        parts.append(
            f'    <text\n'
            f'      x="{lx:.2f}" y="{ly:.2f}"\n'
            f'      text-anchor="middle"\n'
            f'      dominant-baseline="central"\n'
            f'      font-family="{LABEL_FONT_FAMILY}"\n'
            f'      font-weight="{LABEL_FONT_WEIGHT}"\n'
            f'      font-size="{fs}"\n'
            f'      fill="{LABEL_COLOR}"\n'
            f'      data-region-id="{region.region_id}"\n'
            f'    >{number}</text>\n'
        )

    parts.append('  </g>\n')
    parts.append('</svg>')
    return "".join(parts)


def _build_animated_svg(
    regions: List[Region],
    region_paths: Dict[int, str],
    w: int,
    h: int,
) -> str:
    """
    Build an SVG that animates from outline (white) to colored.
    Staggers the fill animation across regions for a pleasing effect.
    """
    parts = [_svg_header(w, h)]
    parts.append(_background_rect(w, h, BACKGROUND_COLOR))
    parts.append('  <g id="regions">\n')

    # Total duration of the reveal in seconds
    total_reveal_time = 3.0
    num_regions = len(regions)

    for i, region in enumerate(regions):
        path_d = region_paths.get(region.region_id)
        if not path_d:
            continue

        # Stagger the start time based on region index or area
        delay = (i / max(1, num_regions)) * total_reveal_time
        dur = 0.8  # duration of individual fill transition

        parts.append(
            f'    <path\n'
            f'      id="region-{region.region_id}"\n'
            f'      d="{path_d}"\n'
            f'      fill="{OUTLINE_FILL}"\n'
            f'      stroke="{STROKE_COLOR}"\n'
            f'      stroke-width="{STROKE_WIDTH}"\n'
            f'      stroke-linejoin="round"\n'
            f'      fill-rule="evenodd"\n'
        )
        # Add SMIL animation
        parts.append(
            f'    >\n'
            f'      <animate\n'
            f'        attributeName="fill"\n'
            f'        from="{OUTLINE_FILL}"\n'
            f'        to="{region.color_hex}"\n'
            f'        dur="{dur}s"\n'
            f'        begin="{delay:.2f}s"\n'
            f'        fill="freeze"\n'
            f'      />\n'
            f'    </path>\n'
        )

    parts.append('  </g>\n')
    parts.append('</svg>')
    return "".join(parts)


def build_palette_legend_svg(
    palette_hex: List[str],
    cell_size: int = 40,
    cols: int = 6,
) -> str:
    """
    Build a small palette legend SVG: colored swatches with color numbers.
    Useful for debug / reference display.
    """
    n = len(palette_hex)
    rows = (n + cols - 1) // cols
    w = cols * cell_size
    h = rows * cell_size

    parts = [_svg_header(w, h)]
    parts.append(_background_rect(w, h, "#ffffff"))

    for i, hex_color in enumerate(palette_hex):
        col = i % cols
        row = i // cols
        x = col * cell_size
        y = row * cell_size

        parts.append(
            f'  <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
            f'fill="{hex_color}" stroke="#333" stroke-width="0.5"/>\n'
        )
        parts.append(
            f'  <text x="{x + cell_size//2}" y="{y + cell_size//2}" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-size="12" font-weight="bold" fill="#000">{i+1}</text>\n'
        )

    parts.append('</svg>')
    return "".join(parts)
