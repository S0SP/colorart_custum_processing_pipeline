"""
Standalone pipeline test — no server required.
Run: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import numpy as np
from PIL import Image
import io

def make_test_image() -> bytes:
    """Generate a synthetic color-block image for testing."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Color blocks
    img[0:200, 0:200]   = [255, 100, 100]   # red zone
    img[0:200, 200:400] = [100, 200, 100]   # green zone
    img[0:200, 400:600] = [100, 100, 255]   # blue zone
    img[200:400, 0:300] = [255, 255, 100]   # yellow zone
    img[200:400, 300:600] = [200, 100, 255] # purple zone
    # Add a hole-like circle in the red zone
    import cv2
    cv2.circle(img, (100, 100), 40, (255, 200, 50), -1)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_with_real_image(image_path: str) -> dict:
    from pipeline.orchestrator import process_image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return process_image(image_bytes, num_colors=12)


def test_synthetic() -> dict:
    from pipeline.orchestrator import process_image
    image_bytes = make_test_image()
    return process_image(image_bytes, num_colors=6, max_dimension=600)


if __name__ == "__main__":
    print("=" * 60)
    print("Color-by-Number Pipeline Test")
    print("=" * 60)

    # ── Synthetic test ───────────────────────────────────
    print("\n[1] Running synthetic image test...")
    t0 = time.perf_counter()
    result = test_synthetic()
    elapsed = time.perf_counter() - t0

    print(f"  Total time : {elapsed:.3f}s")
    print(f"  Pipeline   : {result['timing']}")
    print(f"  Image size : {result['width']}x{result['height']}")
    print(f"  Regions    : {result['meta']['num_regions']}")
    print(f"  Palette    : {result['palette']}")
    print(f"  Is illus.  : {result['meta']['is_illustration']}")

    # Save SVGs for synthetic test
    with open("synthetic_colored.svg", "w") as f:
        f.write(result["svg_colored"])
    with open("synthetic_outline.svg", "w") as f:
        f.write(result["svg_outline"])
    with open("synthetic_animated.svg", "w") as f:
        f.write(result["svg_animated"])
    print(f"\n  ✓ Saved synthetic SVGs to current directory")

    # Check a few regions
    print(f"\n  Sample regions:")
    for r in result["regions"][:3]:
        print(f"    Region {r['region_id']}: color #{r['color_number']} "
              f"{r['color_hex']}, area={r['area']}, "
              f"label=({r['label_x']:.1f}, {r['label_y']:.1f}) "
              f"font={r['label_font_size']}px")

    # Check adjacency
    print(f"\n  Sample adjacency (first 3 nodes):")
    for rid, neighbors in list(result["adjacency"].items())[:3]:
        print(f"    Region {rid} → {neighbors}")

    # Validate SVG content
    assert "<svg" in result["svg_colored"], "Missing colored SVG"
    assert "<svg" in result["svg_outline"], "Missing outline SVG"
    assert "<animate" in result["svg_animated"], "Missing animations in animated SVG"
    assert "<text" in result["svg_outline"], "Missing number labels in outline SVG"
    assert "<path" in result["svg_outline"], "Missing paths in outline SVG"
    print("\n  ✓ SVG validation passed")

    # ── Real image test ──────────────────────────────────
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"\n[2] Running real image test: {path}")
        t0 = time.perf_counter()
        result2 = test_with_real_image(path)
        elapsed2 = time.perf_counter() - t0
        print(f"  Total time : {elapsed2:.3f}s")
        print(f"  Regions    : {result2['meta']['num_regions']}")
        print(f"  Timing     : {result2['timing']}")

        # Save SVGs
        out_dir = os.path.dirname(os.path.abspath(path))
        colored_path = os.path.join(out_dir, "output_colored.svg")
        outline_path = os.path.join(out_dir, "output_outline.svg")
        animated_path = os.path.join(out_dir, "output_animated.svg")
        with open(colored_path, "w") as f:
            f.write(result2["svg_colored"])
        with open(outline_path, "w") as f:
            f.write(result2["svg_outline"])
        with open(animated_path, "w") as f:
            f.write(result2["svg_animated"])
        print(f"  Saved SVGs to {out_dir}")

    print("\n✅ All tests passed!")
