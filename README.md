# Color-by-Number Backend Pipeline

> **Image → Clean SVG + Region Map + Color Palette + Adjacency Graph**  
> Fully algorithmic. No GPU. No heavy ML models. ~1–2s per image.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    POST /api/process                         │
│              (multipart: image + params)                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 1: Load + Validate       │
          │   PIL decode → RGB numpy array   │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 2: Resize               │
          │   Longest side ≤ max_dimension  │
          │   (default 1024px)              │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 3: Illustration detect  │
          │   Laplacian variance + color    │
          │   density heuristic             │
          │   → skip bilateral for cartoons │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 4: Color Quantization   │  quantizer.py
          │                                 │
          │   MiniBatchKMeans(k=N_colors)   │
          │   → N-color label_map H×W       │
          │   → micro-region merging        │
          │   → palette (N,3) uint8         │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 5: Region Segmentation  │  segmenter.py
          │                                 │
          │   Per color: scipy.label()      │
          │   → connected components        │
          │   → outer contour + holes       │
          │   → Douglas-Peucker simplify    │
          │   → Region dataclass list       │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 6: Label Placement      │  labeler.py
          │                                 │
          │   Per region:                   │
          │   distance_transform_edt(mask)  │
          │   → pole of inaccessibility     │
          │   → (x, y, font_size)           │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 7: Adjacency Graph      │  adjacency.py
          │                                 │
          │   Per region: dilate mask 2px   │
          │   → ring = dilated - original   │
          │   → scan id_map at ring pixels  │
          │   → {region_id: [neighbors]}    │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   Stage 8: SVG Generation       │  vectorizer.py
          │                                 │  svg_builder.py
          │   Per contour:                  │
          │   1. Chaikin corner-cutting ×3  │
          │   2. Catmull-Rom → Cubic Bezier │
          │   3. SVG "C" commands           │
          │                                 │
          │   → svg_colored (filled)        │
          │   → svg_outline (white+numbers) │
          └────────────────┬────────────────┘
                           │
          ┌────────────────▼────────────────┐
          │   JSON Response                  │
          └─────────────────────────────────┘
```

---

## File Structure

```
colorbynumber_backend/
├── main.py                    # FastAPI app & endpoints
├── requirements.txt
├── Dockerfile
├── pipeline/
│   ├── __init__.py
│   ├── orchestrator.py        # Pipeline runner (calls all stages)
│   ├── quantizer.py           # Color quantization (KMeans)
│   ├── segmenter.py           # Connected component extraction
│   ├── vectorizer.py          # Contour → smooth Bezier SVG paths
│   ├── labeler.py             # Pole of inaccessibility label placement
│   ├── adjacency.py           # Region adjacency graph
│   └── svg_builder.py         # Final SVG assembly
└── tests/
    └── test_pipeline.py       # Standalone test (no server needed)
```

---

## Key Algorithms Explained

### 1. Color Quantization — `quantizer.py`
Uses **MiniBatchKMeans** (scikit-learn). MiniBatch variant is 3–10× faster
than standard KMeans with nearly identical quality for this use case.

**Micro-region cleanup**: After quantization, any connected component
< 0.1% of total pixels is merged into its dominant neighbor using
morphological dilation. Eliminates noise speckle before contour extraction.

### 2. Contour Smoothing — `vectorizer.py`
Two-stage smoothing:

**Stage A — Chaikin Corner Cutting (3 iterations)**  
Chaikin's algorithm is a corner-cutting subdivision that turns a jagged
polyline into a quadratic B-spline approximation. After 3 iterations,
point count multiplies by 8× but the curve is very smooth.

```
Iteration 1:  A──B──C  →  A──q1──r1──q2──r2──q3──r3──C
              where qᵢ = 0.75·Pᵢ + 0.25·Pᵢ₊₁
                    rᵢ = 0.25·Pᵢ + 0.75·Pᵢ₊₁
```

**Stage B — Catmull-Rom → Cubic Bézier**  
Catmull-Rom splines pass through every control point (unlike pure Bézier)
producing natural-looking curves. Each segment P[i]→P[i+1] maps to SVG
cubic Bézier `C CP1 CP2 P[i+1]` where:
```
CP1 = P[i]   + (P[i+1] - P[i-1]) / 6
CP2 = P[i+1] - (P[i+2] - P[i])   / 6
```
This ensures C¹ continuity (tangent continuity) at every junction.

### 3. Pole of Inaccessibility — `labeler.py`
The number inside each region is placed at the **pole of inaccessibility**:
the point inside the region that is furthest from any boundary.
This is equivalent to the center of the largest inscribed circle.

Implementation:
```python
edt = scipy.ndimage.distance_transform_edt(region_mask)
# edt[y,x] = distance from pixel (y,x) to nearest boundary
best_y, best_x = argmax(edt)
```

For irregular shapes (like a crescent), this naturally picks the widest part.
Among all pixels at ≥92% of the max EDT value, we pick the one closest to
the centroid — this avoids choosing a point in a narrow appendage.

**Font size** is proportional to the inscribed circle radius so numbers
always fit visually.

### 4. Adjacency Graph — `adjacency.py`
For each region mask:
1. Dilate by 2 pixels
2. Compute ring = dilated − original
3. Look up all unique region IDs in the `id_map` at ring positions
4. Those IDs are adjacent

This is O(N·H·W) — one dilation pass per region — but in practice very fast
because dilation only affects boundary pixels.

### 5. SVG Holes — `svg_builder.py`
Uses **even-odd fill rule** (`fill-rule="evenodd"`) combined with reversed
winding for hole contours. The outer contour goes clockwise; hole contours
go counter-clockwise. The SVG engine fills the outer shape and punches
holes correctly — no clip paths needed.

---

## API Reference

### `POST /api/process`

**Form fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | File | required | JPEG / PNG / WebP / BMP |
| `num_colors` | int | 12 | Palette size (4–32) |
| `max_dimension` | int | 1024 | Resize longest side to this |
| `min_region_area` | int | 80 | Drop regions smaller than N px |

**Response:**

```json
{
  "width": 1024,
  "height": 768,
  "svg_colored": "<svg>...</svg>",
  "svg_outline": "<svg>...</svg>",
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
    }
  ],
  "palette": ["#ff6b9d", "#4a90e2", "..."],
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

---

## SVG Element Structure

### Outline SVG (for the Android app canvas)
```xml
<svg viewBox="0 0 1024 768" ...>
  <rect width="1024" height="768" fill="#ffffff"/>

  <!-- Regions: white fill, black outline, even-odd for holes -->
  <g id="regions">
    <path
      id="region-1"
      d="M 100.000,200.000 C ... Z"
      fill="#ffffff"
      stroke="#1a1a1a"
      stroke-width="1.2"
      stroke-linejoin="round"
      fill-rule="evenodd"
      data-region-id="1"
      data-color-number="3"
      data-color="#ff6b9d"
    />
    ...
  </g>

  <!-- Number labels: centered at pole of inaccessibility -->
  <g id="labels">
    <text
      x="140.50" y="230.00"
      text-anchor="middle"
      dominant-baseline="central"
      font-family="Arial, Helvetica, sans-serif"
      font-weight="bold"
      font-size="14"
      fill="#222222"
      data-region-id="1"
    >3</text>
    ...
  </g>
</svg>
```

---

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run dev server
uvicorn main:app --reload --port 8000

# Run tests (no server needed)
python tests/test_pipeline.py

# Test with real image
python tests/test_pipeline.py path/to/your/image.jpg

# Docker (Alternative)
docker build -t cbn-backend .
docker run -p 8000:8000 cbn-backend

## Deploy to Render.com (Native Python)

1.  **Create a New Web Service**: In Render's dashboard, select **New > Web Service**.
2.  **Connect Repo**: Connect your GitHub/GitLab repository.
3.  **Basic Settings**:
    - **Runtime**: `Python`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4.  **Environment Variables**:
    - `PYTHON_VERSION`: `3.11.9`
    - `PORT`: `10000` (Render's default) or any other.
5.  **Click "Deploy"**.

Alternatively, you can just click **"Blueprint"** and Render will automatically use the `render.yaml` file I've added to this repository.
```

---

## Performance

| Stage | Time (1024px illustration) |
|-------|---------------------------|
| Load + resize | ~15ms |
| Color quantization | ~300–500ms |
| Segmentation | ~80–120ms |
| Label placement | ~150–250ms |
| Adjacency graph | ~200–350ms |
| SVG generation | ~100–200ms |
| **Total** | **~850ms–1.4s** |

**Optimization tips:**
- Reduce `max_dimension` to 768 → ~2× speedup
- Reduce `num_colors` to 8 → ~30% speedup
- **Gzip Compression**: Automatically enabled in `main.py` for SVGs > 1KB.
- **Environment Variables**: Use a `.env` file (see `.env.example`).
- **Concurrent Workers**: The Dockerfile is configured with 2 workers. Adjust based on CPU cores.
- **Reverse Proxy**: Always run behind Nginx/Caddy with SSL.
- **Auto-Extraction**: Processing results are auto-saved to `./output/` for debugging. Ensure this directory is mounted if using Docker.

---

## Android Client Integration

```kotlin
// Kotlin / Retrofit example
interface ColorByNumberApi {
    @Multipart
    @POST("api/process")
    suspend fun processImage(
        @Part image: MultipartBody.Part,
        @Part("num_colors") numColors: RequestBody,
        @Part("max_dimension") maxDim: RequestBody,
    ): ProcessResponse
}

data class ProcessResponse(
    val width: Int,
    val height: Int,
    val svgOutline: String,     // load into WebView or SVG renderer
    val svgColored: String,
    val regions: List<RegionDto>,
    val palette: List<String>,
    val adjacency: Map<String, List<Int>>
)
```

**Rendering the SVG on Android:**
- Use **[AndroidSVG](https://bigbadaboom.github.io/androidsvg/)** library to render SVG to Canvas
- Or embed in a `WebView` with touch hit-testing on `<path>` elements
- Tap detection: use `SVGImageView.getTag()` on `data-region-id` to identify which region was tapped
