# 👑 Color-by-Number API & Frontend Beast-Performance Guide

This guide is designed for **LLM Orchestrators** and **Frontend Engineers** to build a top-tier (50M+ downloads) coloring game using this backend.

---

## 📡 1. The API Endpoint
**URL**: `POST /api/process`  
**Content-Type**: `multipart/form-data`

### 📥 Request Parameters
| Parameter | Type | Range | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `image` | File | <20MB | - | Source image (JPG, PNG, WebP). |
| `num_colors` | int | 4–128 | 32 | **Color Depth**: Higher = more realistic shading. |
| `target_regions`| int | 50–5000 | - | **Complexity**: Direct control over shape count. |
| `max_dimension` | int | 256–2048 | 1024 | **Resolution**: 2048 recommended for mandalas. |
| `min_region_area`| int | 1–5000 | 25 | **Noise Gate**: Set to 1 for extreme detail. |

---

## 📤 2. The JSON Response Breakdown

### 🖼️ Global Metadata
- `width`, `height`: Processing dimensions (use for canvas aspect ratio).
- `thumbnail_b64`: **Instant Gallery**. A small JPG string to show levels instantly without loading full SVGs.
- `palette`: List of hex codes (e.g., `["#FF0000", ...]`).

### ⚡ Mega-Paths (Performance King)
- `mega_paths_by_color`: A dictionary where keys are color indices and values are **long SVG path strings**. 
  - *RECIPE*: Render these once as a background layer. It is **100x faster** than rendering 4,000 separate regions.

### 🧩 Interactive Regions Layer
The `regions` array contains objects with:
- `region_id`: Unique ID for tracking (e.g., "was this shape tapped?").
- `color_idx`: Reference to the palette index.
- `path_data`: The SVG path string for **just this shape**.
- `label_x`, `label_y`: Optimized coordinates for the number label.
- `label_font_size`: Pre-calculated size (Pole of Inaccessibility method).
- `hint_priority`: **0-100 score**. 100 = tiny/hidden. 0 = huge/obvious.

---

## 🚀 3. Frontend Beast-Performance Recipes

### 🔍 A. Pro Zoom & Pan (60 FPS)
1. Use `react-native-svg` with `react-native-gesture-handler`.
2. **Layering**: 
   - **Layer 0 (Base)**: Render the `mega_paths_by_color`.
   - **Layer 1 (Interaction)**: Map through `regions` but only render `Path` segments that are in the user's current view (Culling).
3. **Re-center Logic**: 
   - If User moves the canvas more than 50px away from the center or zooms past 1.0x, show a "Center" button in the bottom right.
   - When tapped, use `Animated.spring()` to reset `translation` and `scale` to center.

### 💡 B. The Smart Hint System
1. Filter out already colored regions.
2. Sort remaining regions by `hint_priority` (Descending).
3. Take the #1 region and use `label_x/y` to move the camera there.
4. Add a "Pulse" animation to that specific `Path`.

### 🏁 C. 100% Completion (Flood Fill Animation)
When the user finishes the last shape:
1. Use the `adjacency` map to trigger a "Wave of Color."
2. Start from the center region, then animate all neighbors, then neighbors of neighbors.
3. This creates a satisfying ripple effect that feels premium.

### 📍 D. Perfect Number Centering
- The backend uses the **"Pole of Inaccessibility"** (not simple 
area center) to ensure labels stay inside wavy/skinny shapes.
- Frontend: Set `textAnchor="middle"` and `alignmentBaseline="middle"` to keep them perfectly locked inside boundaries.

---

## 🎨 4. Style Logic (For LLM Prompting)

- **Cartoon/Kids**: `num_colors: 12, target_regions: 300, min_region_area: 80`
- **Modern Flat**: `num_colors: 24, target_regions: 800, min_region_area: 25`
- **Oil Painting**: `num_colors: 64, target_regions: 2000, min_region_area: 10`
- **Master Mandala**: `num_colors: 8, target_regions: 4000, min_region_area: 1, max_dimension: 2048`
