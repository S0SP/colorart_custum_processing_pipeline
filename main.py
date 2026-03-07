"""
Color-by-Number Backend API
============================
FastAPI server exposing the image → SVG pipeline.

Endpoints:
  POST /api/process
    - Input:  multipart image file + optional JSON params
    - Output: JSON with svg_outline, svg_colored, regions, palette, adjacency

  GET  /api/health
    - Liveness check

  GET  /docs
    - Auto-generated Swagger UI

Run:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from pipeline.orchestrator import process_image

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Production Toggle: Set DEBUG_MODE=true in .env or environment to save files to disk
DEBUG_SAVING = os.getenv("DEBUG_MODE", "False").lower() == "true"

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Color-by-Number Pipeline API",
    description=(
        "Converts any image to a color-by-number SVG with region map, "
        "palette, adjacency graph, and centered number labels."
    ),
    version="1.1.0",
)

# ── CORS (allow Android/web clients) ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.get("/")
async def root():
    return {
        "message": "ColorArt Backend is running!",
        "documentation": "/docs",
        "health": "/api/health"
    }


# ─────────────────────────────────────────────────────────
#  Health check
# ─────────────────────────────────────────────────────────
@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok", "version": app.version}


# ─────────────────────────────────────────────────────────
#  Main processing endpoint
# ─────────────────────────────────────────────────────────
@app.post("/api/process", tags=["pipeline"])
async def process(
    image: UploadFile = File(..., description="Source image (JPEG, PNG, WebP, etc.)"),
    num_colors: Optional[int] = Form(
        default=32,
        description="Number of palette colors (4–128). Default 32.",
        ge=4,
        le=128,
    ),
    max_dimension: Optional[int] = Form(
        default=1024,
        description="Longest side resized to this (256–2048). Default 1024.",
        ge=256,
        le=2048,
    ),
    min_region_area: Optional[int] = Form(
        default=25,
        description="Minimum region size in pixels. Default 25.",
        ge=1,
        le=10000,
    ),
    target_regions: Optional[int] = Form(
        default=None,
        description="Target number of regions. If set, merges smallest regions until reached.",
        ge=10,
        le=5000,
    ),
):
    """
    Process an image through the Color-by-Number pipeline.

    ### Response shape
    ```json
    {
      "width": 1024,
      "height": 768,
      "svg_colored": "<svg>...</svg>",
      "svg_outline": "<svg>...</svg>",
      "svg_animated": "<svg>...</svg>",
      "svg_palette_legend": "<svg>...</svg>",
      "regions": [
        {
          "region_id": 1,
          "color_number": 3,
          "color_hex": "#ff6b9d",
          "color_rgb": [255, 107, 157],
          "area": 5234,
          "bbox": {"x": 100, "y": 200, "w": 80, "h": 60},
          "label_x": 140.5,
          "label_y": 230.0,
          "label_font_size": 14
        },
        ...
      ],
      "palette": ["#ff6b9d", "#4a90e2", ...],
      "adjacency": {
        "1": [2, 5, 7],
        "2": [1, 3]
      },
      "timing": {
        "load": 0.012,
        "quantize": 0.345,
        "segment": 0.089,
        "labels": 0.156,
        "adjacency": 0.234,
        "svg": 0.123,
        "total": 0.959
      },
      "meta": {
        "num_colors_requested": 12,
        "num_regions": 87,
        "is_illustration": true
      }
    }
    ```
    """
    # ── Validate content type ──────────────────────────────
    allowed_types = {
        "image/jpeg", "image/jpg", "image/png",
        "image/webp", "image/bmp", "image/tiff",
    }
    ct = (image.content_type or "").lower()
    if ct not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ct}'. Accepted: {', '.join(sorted(allowed_types))}",
        )

    # ── Read file ─────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB hard limit
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    logger.info(
        f"Processing '{image.filename}' — "
        f"{len(image_bytes)/1024:.1f} KB, "
        f"num_colors={num_colors}, max_dim={max_dimension}"
    )

    # ── Run pipeline ───────────────────────────────────────
    try:
        result = process_image(
            image_bytes=image_bytes,
            num_colors=num_colors,
            max_dimension=max_dimension,
            min_region_area=min_region_area,
            target_regions=target_regions,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    logger.info(
        f"Done — {result['meta']['num_regions']} regions, "
        f"total={result['timing']['total']:.3f}s"
    )

    # ── Automatic Extraction (Debugging) ──────────────────
    if DEBUG_SAVING:
        try:
            from pathlib import Path
            out_root = Path("output")
            out_root.mkdir(exist_ok=True)
            
            # Clean base name
            base_name = "".join(c for c in Path(image.filename).stem if c.isalnum() or c in (' ', '.', '_')).rstrip()
            if not base_name:
                base_name = "upload"
                
            # Find unique name with counter
            counter = 1
            final_name = f"{base_name}_{counter}"
            while (out_root / f"{final_name}_animated.svg").exists():
                counter += 1
                final_name = f"{base_name}_{counter}"
            
            # Save SVGs
            (out_root / f"{final_name}_animated.svg").write_text(result["svg_animated"], encoding="utf-8")
            (out_root / f"{final_name}_outline.svg").write_text(result["svg_outline"], encoding="utf-8")
            (out_root / f"{final_name}_colored.svg").write_text(result["svg_colored"], encoding="utf-8")
            
            # Save Debug Info
            debug_path = out_root / f"{final_name}_debug.json"
            with open(debug_path, "w") as f:
                import json
                debug_data = {k: v for k, v in result.items() if not k.startswith("svg_")}
                json.dump(debug_data, f, indent=2)
                
            logger.info(f"Automatically saved results to {out_root.absolute()} as {final_name}")
        except Exception as save_err:
            logger.warning(f"Failed to auto-save results: {save_err}")

    return JSONResponse(content=result)


# ─────────────────────────────────────────────────────────
#  Dev entrypoint
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
        workers=1,
    )
