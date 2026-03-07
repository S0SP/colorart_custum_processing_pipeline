# 👑 Color-by-Number Elite Master Guide

This document is the ultimate reference for building a top-tier (50M+ downloads) coloring game using this backend. It covers every byte of data provided by the API and the exact logic to implement "Triple-A" game mechanics in React Native.

---

## 📡 1. API Architecture (The "Take & Give")

### 📥 The Request: `POST /api/process`
**Content-Type**: `multipart/form-data`

| Parameter | Type | Default | Logic for 50M+ Downloads |
| :--- | :--- | :--- | :--- |
| `image` | File | - | Use high-quality sources (Midjourney/DALL-E). |
| `num_colors` | int | 32 | **Detail Scaling**: Use 12 for "Kids Mode", 64 for "Art Mode", 128 for "Photo Mode". |
| `target_regions` | int | None | **Game Length**: Set 200 for a 2-min level, 3000 for a 30-min challenge. |
| `max_dimension` | int | 1024 | **Edge Quality**: Always use 2048 for Mandalas or fine line art. |
| `min_region_area` | int | 25 | **Difficulty Control**: Set to 1 to preserve every tiny dot; set to 100 for big chunky shapes. |

### 📤 The Response (Hyper-Detailed JSON)
The backend returns a deeply structured object designed for zero-latency frontend work.

```json
{
  "width": 1024, "height": 768,
  "thumbnail_b64": "<base64_jpg>",
  "mega_paths_by_color": { "0": "M 0 0...", "1": "M 10 20..." },
  "regions": [
     {
       "region_id": 402,
       "color_idx": 3,
       "color_hex": "#FF5733",
       "path_data": "M 100 200 L 105 205...",
       "label_x": 102.5, "label_y": 202.3,
       "label_font_size": 14.5,
       "hint_priority": 98
     }
  ],
  "palette_stats": [
     { "color_idx": 0, "area_fraction": 0.25, "region_count": 45 }
  ],
  "adjacency": { "402": [403, 405, 510] }
}
```

---

## ⚡ 2. High-Performance Frontend Rendering

### 🏎️ Tiered Layering Strategy (The 60FPS Secret)
Do **not** render all paths as interactive elements. Use three distinct layers:

1.  **LAYER 1: The "Ghost" Base (Background)**:
    - Render the `mega_paths_by_color` with `fill="#EEE"`. This shows the user the gray silhouette of the image.
2.  **LAYER 2: The "Fill" Layer (Performance)**:
    - When a user finishes Color #5, do **not** re-render 500 small paths.
    - Instead, render the single `mega_path` for Color #5. This is 100x faster for the GPU.
3.  **LAYER 3: The "Tap" Layer (Active Color)**:
    - Only render individual `regions` for the **currently selected color number**.
    - If user selects Color #3, only Color #3's shapes are clickable. This keeps the DOM/Memory clean.

---

## 🎮 3. Elite Game Mechanics

### 📊 A. The "Gold Coin" Progress Bar
**Requirement**: A top status bar that fills up as you play.
- **Backend Usage**: Use `palette_stats`. 
- **Recipe**:
  - `total_weight = 100` (The full width of the bar).
  - Each time a region is colored: `progress += (region.area / total_image_area)`.
  - **The "Gold" Effect**: When the progress bar hits 100%, use `react-native-confetti-cannon` or a particle system to spray **Gold Coins** from the top of the screen.

### 🔦 B. Highlight Selected Color
**Requirement**: "Where are all the #3 spots?"
- **Recipe**: On the frontend, when Color #3 is selected in the palette:
  - Increase the `opacity` of the `mega_path` for Color #3.
  - Add a **subtle pulse animation** (using `Reanimated`) to all uncolored shapes belonging to Color #3.

### 💡 C. Smart Hint System (The Frustration Killer)
**Requirement**: High-priority hints.
- **Logic**: The backend sorts regions by `hint_priority`. 
- **Recipe**:
  1. Find the non-colored region with the **highest** `hint_priority` (usually the smallest, hardest-to-find ones).
  2. Use `label_x` and `label_y` to center the camera.
  3. Highlight that exact `path_data` with a bright "Glow" effect.

### 🏁 D. 100% Completion: Adjacency Wave
**Requirement**: The "Flood Fill" satisfaction animation.
- **Logic**: When the last shape is tapped, trigger a chain reaction using the `adjacency` map.
- **Recipe**:
  - `Step 0`: Color the final shape.
  - `Step 1`: Find all neighbor IDs in the `adjacency` map for that shape.
  - `Step 2`: Trigger a "Flash" animation on those neighbors.
  - `Step 3`: Repeat for the neighbors' neighbors until the whole image "pulses" in a color wave.

---

## � 5. Sensory Experience (Haptics & Sound)

To hit 50M+ downloads, the game must feel "crunchy" and responsive.

### 📳 A. Haptic Feedback (Vibration)
**Requirement**: Physical response to user errors.
- **Recipe**:
  - Use `react-native-haptic-feedback`.
  - **Wrong Region**: When a user taps a region that doesn't match the selected color number, trigger a **"notificationError"** or **"heavy"** vibration pattern. This instantly tells the user "Nope!" without needing a popup.

### 🎵 B. The "Audio Rewards" System
**Requirement**: Satisfying sounds for progression.
- **1. The "Squash" (Individual Fill)**:
  - Play a short, high-quality "plop" or "squash" sound every time a single region is correctly filled. 
  - *Pro Tip*: Slightly randomize the pitch of the sound (±5%) so it doesn't get annoying after 1000 taps.
- **2. The "Chime" (Color Completion)**:
  - When the *last* region of a specific color is filled (Check `regions` array for that `color_idx`), play a triumphant "Ding!" or "Glitter" sound.
  - Visual: Animate the palette circle for that color to shrink and reveal a **Checkmark** or **Gold Star**.
- **3. The "Jackpot" (Level Complete)**:
  - Trigger a long, cascading "fanfare" sound alongside the **Adjacency Wave (Section 3D)**.

---

## �🔍 4. Precision Zoom & Navigation

## � 4. Precision Zoom & Navigation

### 🔍 A. Pro Zoom & Pan (60 FPS)
1. Use `react-native-svg` with `react-native-gesture-handler`.
2. **Layering**: 
   - **Layer 0 (Base)**: Render the `mega_paths_by_color`.
   - **Layer 1 (Interaction)**: Map through `regions` but only render `Path` segments that are in the user's current view (Culling).
3. **Adaptive Number Visibility (Zoom Effect)**:
   - **Logic**: Numbers in tiny regions clutter the screen. Only show them when they are "big enough" to read.
   - **Recipe**:
     - Calculated Visibility: `isVisible = (region.area * currentZoom) > THRESHOLD`.
     - Set the `opacity` of the `<Text>` label based on this boolean.
     - **Exception**: If the color of that region is the **Currently Selected Color** or if a **Hint** is active, force `opacity = 1` even if small.
4. **Re-center Logic**: 
   - If User moves the canvas more than 50px away from the center or zooms past 1.0x, show a "Center" button in the bottom right.
   - When tapped, use `Animated.spring()` to reset `translation` and `scale` to center.

### 🖼️ Instant-Load Thumbnails
- **Why**: High-region SVGs can be 5MB+.
- **How**: Store the `thumbnail_b64` in your database. 
- **UX**: Show the thumbnail instantly on your "Level Select" screen. Only download the full 4000-region JSON when the user actually taps "Play."

---

## ⚙️ 5. Production Maintenance
- **`DEBUG_SAVING`**: Keep this `False` in your `.env`.
- **Latency**: The pipeline runs in ~1.5s. Use a "Scanning..." animation on the frontend to keep the user engaged during the wait.

**Your backend is now a military-grade coloring engine. Go build a hit!** 🏆🚀
