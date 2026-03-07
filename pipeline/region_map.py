"""
Region Map Generator for O(1) Tap Detection
============================================

Creates a compressed 2D grid where each pixel contains the region_id
at that location. Frontend decodes this for instant tap detection
instead of expensive path.contains() calls.

Performance: 700 regions → <0.5ms tap detection (vs 50ms with path.contains)
"""

import numpy as np
import cv2
import zlib
import base64
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Point:
    x: float
    y: float


class SVGPathParser:
    """
    Parses SVG path data (d attribute) into polygon points.
    Handles: M, m, L, l, H, h, V, v, C, c, S, s, Q, q, T, t, A, a, Z, z
    """
    
    def __init__(self, path_data: str):
        self.path_data = path_data
        self.points: List[Point] = []
        self.current = Point(0, 0)
        self.start = Point(0, 0)
        self.last_control: Optional[Point] = None
        
    def parse(self) -> List[Tuple[float, float]]:
        """Parse SVG path and return list of (x, y) points."""
        if not self.path_data or not self.path_data.strip():
            return []
            
        self.points = []
        self.current = Point(0, 0)
        self.start = Point(0, 0)
        self.last_control = None
        
        # Tokenize: split into commands and numbers
        tokens = re.findall(
            r'([MmLlHhVvCcSsQqTtAaZz])|([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)',
            self.path_data
        )
        
        i = 0
        current_cmd = None
        
        while i < len(tokens):
            token = tokens[i][0] if tokens[i][0] else tokens[i][1]
            
            if not token:
                i += 1
                continue
            
            # Check if it's a command letter
            if token.isalpha():
                current_cmd = token
                i += 1
                continue
            
            if current_cmd is None:
                i += 1
                continue
                
            # Process based on current command
            try:
                if current_cmd in ('M', 'm'):
                    i = self._handle_moveto(tokens, i, current_cmd == 'm')
                    current_cmd = 'L' if current_cmd == 'M' else 'l'
                    
                elif current_cmd in ('L', 'l'):
                    i = self._handle_lineto(tokens, i, current_cmd == 'l')
                    
                elif current_cmd in ('H', 'h'):
                    i = self._handle_horizontal(tokens, i, current_cmd == 'h')
                    
                elif current_cmd in ('V', 'v'):
                    i = self._handle_vertical(tokens, i, current_cmd == 'v')
                    
                elif current_cmd in ('C', 'c'):
                    i = self._handle_cubic(tokens, i, current_cmd == 'c')
                    
                elif current_cmd in ('S', 's'):
                    i = self._handle_smooth_cubic(tokens, i, current_cmd == 's')
                    
                elif current_cmd in ('Q', 'q'):
                    i = self._handle_quadratic(tokens, i, current_cmd == 'q')
                    
                elif current_cmd in ('T', 't'):
                    i = self._handle_smooth_quadratic(tokens, i, current_cmd == 't')
                    
                elif current_cmd in ('A', 'a'):
                    i = self._handle_arc(tokens, i, current_cmd == 'a')
                    
                elif current_cmd in ('Z', 'z'):
                    self._handle_closepath()
                    i += 1
                    current_cmd = None
                    
                else:
                    i += 1
            except (IndexError, ValueError) as e:
                # Skip malformed segments
                i += 1
                continue
        
        return [(p.x, p.y) for p in self.points]
    
    def _get_number(self, tokens: List, index: int) -> Tuple[float, int]:
        """Extract a number from tokens at given index."""
        if index >= len(tokens):
            return 0.0, index + 1
        token = tokens[index][1] if tokens[index][1] else tokens[index][0]
        if not token or token.isalpha():
            return 0.0, index
        try:
            return float(token), index + 1
        except (ValueError, TypeError):
            return 0.0, index + 1
    
    def _add_point(self, x: float, y: float):
        """Add a point and update current position."""
        self.current = Point(x, y)
        self.points.append(Point(x, y))
    
    def _handle_moveto(self, tokens: List, i: int, relative: bool) -> int:
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x += self.current.x
            y += self.current.y
        
        self.current = Point(x, y)
        self.start = Point(x, y)
        self.points.append(Point(x, y))
        self.last_control = None
        return i
    
    def _handle_lineto(self, tokens: List, i: int, relative: bool) -> int:
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x += self.current.x
            y += self.current.y
        
        self._add_point(x, y)
        self.last_control = None
        return i
    
    def _handle_horizontal(self, tokens: List, i: int, relative: bool) -> int:
        x, i = self._get_number(tokens, i)
        
        if relative:
            x += self.current.x
        
        self._add_point(x, self.current.y)
        self.last_control = None
        return i
    
    def _handle_vertical(self, tokens: List, i: int, relative: bool) -> int:
        y, i = self._get_number(tokens, i)
        
        if relative:
            y += self.current.y
        
        self._add_point(self.current.x, y)
        self.last_control = None
        return i
    
    def _handle_cubic(self, tokens: List, i: int, relative: bool) -> int:
        """Cubic Bezier: C x1 y1 x2 y2 x y"""
        x1, i = self._get_number(tokens, i)
        y1, i = self._get_number(tokens, i)
        x2, i = self._get_number(tokens, i)
        y2, i = self._get_number(tokens, i)
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x1 += self.current.x; y1 += self.current.y
            x2 += self.current.x; y2 += self.current.y
            x += self.current.x; y += self.current.y
        
        self._sample_cubic_bezier(self.current.x, self.current.y, x1, y1, x2, y2, x, y)
        self.last_control = Point(x2, y2)
        return i
    
    def _handle_smooth_cubic(self, tokens: List, i: int, relative: bool) -> int:
        """Smooth cubic: S x2 y2 x y"""
        x2, i = self._get_number(tokens, i)
        y2, i = self._get_number(tokens, i)
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x2 += self.current.x; y2 += self.current.y
            x += self.current.x; y += self.current.y
        
        if self.last_control:
            x1 = 2 * self.current.x - self.last_control.x
            y1 = 2 * self.current.y - self.last_control.y
        else:
            x1, y1 = self.current.x, self.current.y
        
        self._sample_cubic_bezier(self.current.x, self.current.y, x1, y1, x2, y2, x, y)
        self.last_control = Point(x2, y2)
        return i
    
    def _handle_quadratic(self, tokens: List, i: int, relative: bool) -> int:
        """Quadratic Bezier: Q x1 y1 x y"""
        x1, i = self._get_number(tokens, i)
        y1, i = self._get_number(tokens, i)
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x1 += self.current.x; y1 += self.current.y
            x += self.current.x; y += self.current.y
        
        self._sample_quadratic_bezier(self.current.x, self.current.y, x1, y1, x, y)
        self.last_control = Point(x1, y1)
        return i
    
    def _handle_smooth_quadratic(self, tokens: List, i: int, relative: bool) -> int:
        """Smooth quadratic: T x y"""
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x += self.current.x
            y += self.current.y
        
        if self.last_control:
            x1 = 2 * self.current.x - self.last_control.x
            y1 = 2 * self.current.y - self.last_control.y
        else:
            x1, y1 = self.current.x, self.current.y
        
        self._sample_quadratic_bezier(self.current.x, self.current.y, x1, y1, x, y)
        self.last_control = Point(x1, y1)
        return i
    
    def _handle_arc(self, tokens: List, i: int, relative: bool) -> int:
        """Arc: A rx ry x-rotation large-arc sweep x y"""
        rx, i = self._get_number(tokens, i)
        ry, i = self._get_number(tokens, i)
        x_rotation, i = self._get_number(tokens, i)
        large_arc, i = self._get_number(tokens, i)
        sweep, i = self._get_number(tokens, i)
        x, i = self._get_number(tokens, i)
        y, i = self._get_number(tokens, i)
        
        if relative:
            x += self.current.x
            y += self.current.y
        
        # Simplified: approximate arc with quadratic bezier
        self._sample_arc(self.current.x, self.current.y, rx, ry, x_rotation, large_arc, sweep, x, y)
        self.last_control = None
        return i
    
    def _handle_closepath(self):
        """Close path: Z/z"""
        if self.start:
            self._add_point(self.start.x, self.start.y)
    
    def _sample_cubic_bezier(self, x0, y0, x1, y1, x2, y2, x3, y3, num_samples: int = 10):
        """Sample points along a cubic bezier curve."""
        for i in range(1, num_samples + 1):
            t = i / num_samples
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            
            x = mt3 * x0 + 3 * mt2 * t * x1 + 3 * mt * t2 * x2 + t3 * x3
            y = mt3 * y0 + 3 * mt2 * t * y1 + 3 * mt * t2 * y2 + t3 * y3
            
            self._add_point(x, y)
    
    def _sample_quadratic_bezier(self, x0, y0, x1, y1, x2, y2, num_samples: int = 8):
        """Sample points along a quadratic bezier curve."""
        for i in range(1, num_samples + 1):
            t = i / num_samples
            mt = 1 - t
            
            x = mt * mt * x0 + 2 * mt * t * x1 + t * t * x2
            y = mt * mt * y0 + 2 * mt * t * y1 + t * t * y2
            
            self._add_point(x, y)
    
    def _sample_arc(self, x0, y0, rx, ry, rotation, large_arc, sweep, x, y, num_samples: int = 12):
        """Simplified arc sampling."""
        # Approximate with bezier-like curve
        cx = (x0 + x) / 2
        cy = (y0 + y) / 2
        
        dx = x - x0
        dy = y - y0
        
        offset = max(rx, ry) * (0.5 if large_arc else 0.25)
        if sweep:
            ctrl_x = cx - dy * offset / max(abs(dy), 1)
            ctrl_y = cy + dx * offset / max(abs(dx), 1)
        else:
            ctrl_x = cx + dy * offset / max(abs(dy), 1)
            ctrl_y = cy - dx * offset / max(abs(dx), 1)
        
        self._sample_quadratic_bezier(x0, y0, ctrl_x, ctrl_y, x, y, num_samples)


def generate_region_map(
    regions: List[Dict],
    image_width: int,
    image_height: int,
    target_width: int = 512
) -> Dict:
    """
    Generate compressed region ID map for O(1) tap detection.
    
    Args:
        regions: List of region dicts with 'region_id' and 'path_data'
        image_width: Original image width
        image_height: Original image height
        target_width: Target map width (512 = good balance of accuracy vs size)
    
    Returns:
        Dict with region_map_b64, region_map_width, region_map_height, region_map_scale
    """
    if not regions:
        return {
            'region_map_b64': '',
            'region_map_width': 0,
            'region_map_height': 0,
            'region_map_scale': 1.0
        }
    
    # Calculate dimensions
    scale = image_width / target_width
    map_width = target_width
    map_height = max(1, int(image_height / scale))
    
    logger.info(f"[RegionMap] Generating {map_width}x{map_height} map (scale={scale:.2f}) for {len(regions)} regions")
    
    # Create empty region map (0 = no region / background)
    # Using uint16 to support up to 65535 regions
    region_map = np.zeros((map_height, map_width), dtype=np.uint16)
    
    # Sort regions by area (largest first) so smaller regions overlay larger ones
    sorted_regions = sorted(
        regions, 
        key=lambda r: r.get('area', 0), 
        reverse=True
    )
    
    successful = 0
    failed = 0
    
    for region in sorted_regions:
        region_id = region.get('region_id', 0)
        path_data = region.get('path_data', '')
        
        if not path_data or region_id == 0:
            continue
        
        try:
            # Parse SVG path to points
            parser = SVGPathParser(path_data)
            points = parser.parse()
            
            if len(points) < 3:
                failed += 1
                continue
            
            # Scale points to map coordinates
            scaled_points = []
            for x, y in points:
                mx = int(x / scale)
                my = int(y / scale)
                # Clip to valid range
                mx = max(0, min(map_width - 1, mx))
                my = max(0, min(map_height - 1, my))
                scaled_points.append([mx, my])
            
            # Convert to numpy array for OpenCV
            contour = np.array(scaled_points, dtype=np.int32)
            
            # Fill the polygon
            cv2.fillPoly(region_map, [contour], int(region_id))
            successful += 1
            
        except Exception as e:
            failed += 1
            logger.debug(f"[RegionMap] Failed region {region_id}: {e}")
            continue
    
    logger.info(f"[RegionMap] Rasterized {successful}/{len(regions)} regions ({failed} failed)")
    
    # Compress the map
    compressed_data = _compress_region_map(region_map)
    
    return {
        'region_map_b64': compressed_data,
        'region_map_width': map_width,
        'region_map_height': map_height,
        'region_map_scale': round(scale, 4)
    }


def _compress_region_map(region_map: np.ndarray) -> str:
    """
    Compress region map to base64+zlib string.
    Frontend will decode this back to a Uint16Array.
    """
    # Ensure little-endian byte order (matches JavaScript)
    region_map = region_map.astype('<u2')  # little-endian uint16
    
    # Get raw bytes
    raw_bytes = region_map.tobytes()
    
    # Compress with zlib (level 9 = max compression)
    compressed = zlib.compress(raw_bytes, level=9)
    
    # Base64 encode for JSON transport
    b64_data = base64.b64encode(compressed).decode('utf-8')
    
    # Log compression stats
    original_size = len(raw_bytes)
    compressed_size = len(compressed)
    ratio = (1 - compressed_size / original_size) * 100
    
    logger.info(
        f"[RegionMap] Compression: {original_size:,} → {compressed_size:,} bytes "
        f"({ratio:.1f}% reduction), base64: {len(b64_data):,} chars"
    )
    
    return b64_data


def generate_region_map_from_label_map(
    label_map: np.ndarray,
    regions: List,
    target_width: int = 512
) -> Dict:
    """
    Alternative: Generate region map directly from segmentation label_map.
    This is faster than parsing SVG paths but requires access to the label_map.
    
    Args:
        label_map: 2D numpy array where each pixel value is a region label
        regions: List of region objects (need region_id mapping)
        target_width: Target map width
    
    Returns:
        Dict with region_map_b64, region_map_width, region_map_height, region_map_scale
    """
    h, w = label_map.shape[:2]
    scale = w / target_width
    map_width = target_width
    map_height = max(1, int(h / scale))
    
    logger.info(f"[RegionMap] Direct generation: {map_width}x{map_height} from {w}x{h} label_map")
    
    # Resize label map (use NEAREST to preserve label values)
    resized = cv2.resize(
        label_map.astype(np.float32),  # Resize as float to avoid interpolation issues
        (map_width, map_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint16)
    
    # Map label values to region_ids
    # Create lookup: label_value -> region_id
    label_to_region = {}
    for r in regions:
        # Assuming region has original label stored somehow
        # If not, you may need to adjust based on your Region class
        if hasattr(r, 'label_value'):
            label_to_region[r.label_value] = r.region_id
        else:
            # Fallback: assume region_id matches label value
            label_to_region[r.region_id] = r.region_id
    
    # Apply mapping
    region_map = np.zeros_like(resized, dtype=np.uint16)
    for label_val, region_id in label_to_region.items():
        region_map[resized == label_val] = region_id
    
    compressed_data = _compress_region_map(region_map)
    
    return {
        'region_map_b64': compressed_data,
        'region_map_width': map_width,
        'region_map_height': map_height,
        'region_map_scale': round(scale, 4)
    }


# ─── Testing ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    test_regions = [
        {
            'region_id': 1,
            'path_data': 'M 10 10 L 100 10 L 100 100 L 10 100 Z',
            'area': 8100
        },
        {
            'region_id': 2,
            'path_data': 'M 50 50 L 150 50 L 150 150 L 50 150 Z',
            'area': 10000
        },
        {
            'region_id': 3,
            'path_data': 'M 200 200 C 250 200 300 250 300 300 L 200 300 Z',
            'area': 5000
        }
    ]
    
    result = generate_region_map(test_regions, 1024, 768)
    
    print(f"\nResult:")
    print(f"  Width: {result['region_map_width']}")
    print(f"  Height: {result['region_map_height']}")
    print(f"  Scale: {result['region_map_scale']:.2f}")
    print(f"  Data length: {len(result['region_map_b64'])} chars")
    
    # Test decompression
    import base64
    compressed = base64.b64decode(result['region_map_b64'])
    decompressed = zlib.decompress(compressed)
    region_map = np.frombuffer(decompressed, dtype=np.uint16).reshape(
        result['region_map_height'], 
        result['region_map_width']
    )
    print(f"  Unique region IDs: {np.unique(region_map)}")
    print(f"  Region 1 pixels: {np.sum(region_map == 1)}")
    print(f"  Region 2 pixels: {np.sum(region_map == 2)}")
    print(f"  Region 3 pixels: {np.sum(region_map == 3)}")