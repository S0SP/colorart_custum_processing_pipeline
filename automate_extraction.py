"""
Automated SVG Extraction Script
------------------------------
This script processes an image (and a synthetic test) and saves all outputs
(Outline, Colored, and Animated SVGs) into a dedicated 'output/' folder.
"""

import sys
import os
import time
import numpy as np
from PIL import Image
import io
import cv2

# Add current directory to path so we can import pipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.orchestrator import process_image

def make_test_image() -> bytes:
    """Generate a synthetic color-block image for testing."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Color blocks
    img[0:200, 0:200]   = [255, 100, 100]   # red zone
    img[0:200, 200:400] = [100, 200, 100]   # green zone
    img[0:200, 400:600] = [100, 100, 255]   # blue zone
    img[200:400, 0:300] = [255, 255, 100]   # yellow zone
    img[200:400, 300:600] = [200, 100, 255] # purple zone
    cv2.circle(img, (100, 100), 40, (255, 200, 50), -1)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()

def save_outputs(result: dict, prefix: str, output_dir: str):
    """Save colored, outline, and animated SVGs from result dict."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    paths = {
        "colored": os.path.join(output_dir, f"{prefix}_colored.svg"),
        "outline": os.path.join(output_dir, f"{prefix}_outline.svg"),
        "animated": os.path.join(output_dir, f"{prefix}_animated.svg"),
        "debug": os.path.join(output_dir, f"{prefix}_debug.json")
    }
    
    with open(paths["colored"], "w") as f:
        f.write(result["svg_colored"])
    with open(paths["outline"], "w") as f:
        f.write(result["svg_outline"])
    with open(paths["animated"], "w") as f:
        f.write(result["svg_animated"])
        
    # Save debug info as JSON (excluding large SVG strings)
    debug_info = {k: v for k, v in result.items() if not k.startswith("svg_")}
    with open(paths["debug"], "w") as f:
        import json
        json.dump(debug_info, f, indent=2)
        
    return paths

if __name__ == "__main__":
    OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    print("=" * 60)
    print("🚀 Color-by-Number Automation Script")
    print("=" * 60)
    print(f"Target Output Folder: {OUT_DIR}\n")

    # 1. Run Synthetic Test
    print("[1] Processing synthetic test image...")
    sy_bytes = make_test_image()
    sy_result = process_image(sy_bytes, num_colors=6, max_dimension=600)
    sy_paths = save_outputs(sy_result, "synthetic", OUT_DIR)
    print(f"    ✓ Saved animated: {os.path.basename(sy_paths['animated'])}")
    print(f"    ✓ Saved debug info: {os.path.basename(sy_paths['debug'])}")

    # 2. Run Real Image Test (if provided)
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if os.path.exists(img_path):
            print(f"\n[2] Processing real image: {os.path.basename(img_path)}...")
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            
            real_result = process_image(img_bytes, num_colors=12)
            real_paths = save_outputs(real_result, "real", OUT_DIR)
            print(f"    ✓ Saved animated: {os.path.basename(real_paths['animated'])}")
            print(f"    ✓ Saved colored: {os.path.basename(real_paths['colored'])}")
            print(f"    ✓ Saved outline: {os.path.basename(real_paths['outline'])}")
            print(f"    ✓ Saved debug info: {os.path.basename(real_paths['debug'])}")
        else:
            print(f"\n❌ Error: File not found: {img_path}")

    print("\n✨ Automation complete! Check the 'output/' folder.")
