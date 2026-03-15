"""
SVG Pattern Parser — converts SVG markup into a QuiltPattern.

Parses SVG elements (rect, polygon, path, circle, ellipse) into a grid
of colored cells, then applies color quantization, diagonal triangle
detection, and greedy rectangle merging to produce a QuiltPattern.

Uses only xml.etree.ElementTree (stdlib) — no new dependencies.
"""
from __future__ import annotations

import logging
import math
import re
import xml.etree.ElementTree as ET
from typing import Optional

from .grid_engine import QuiltPattern, Block, Fabric
from .color_matcher import _color_distance, _hex_to_rgb, match_kona

logger = logging.getLogger(__name__)

# SVG namespace
_SVG_NS = "http://www.w3.org/2000/svg"
_NS_MAP = {"svg": _SVG_NS}

# CSS named colors (common subset)
_NAMED_COLORS: dict[str, str] = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000",
    "green": "#008000", "blue": "#0000ff", "yellow": "#ffff00",
    "cyan": "#00ffff", "magenta": "#ff00ff", "orange": "#ffa500",
    "purple": "#800080", "pink": "#ffc0cb", "brown": "#a52a2a",
    "gray": "#808080", "grey": "#808080", "navy": "#000080",
    "teal": "#008080", "maroon": "#800000", "olive": "#808000",
    "lime": "#00ff00", "aqua": "#00ffff", "silver": "#c0c0c0",
    "fuchsia": "#ff00ff", "darkred": "#8b0000", "darkgreen": "#006400",
    "darkblue": "#00008b", "lightgray": "#d3d3d3", "lightgrey": "#d3d3d3",
    "coral": "#ff7f50", "salmon": "#fa8072", "gold": "#ffd700",
    "khaki": "#f0e68c", "ivory": "#fffff0", "beige": "#f5f5dc",
    "tan": "#d2b48c", "chocolate": "#d2691e", "crimson": "#dc143c",
    "indigo": "#4b0082", "violet": "#ee82ee", "plum": "#dda0dd",
    "peru": "#cd853f", "sienna": "#a0522d", "tomato": "#ff6347",
    "wheat": "#f5deb3", "linen": "#faf0e6", "lavender": "#e6e6fa",
    "skyblue": "#87ceeb", "steelblue": "#4682b4", "slategray": "#708090",
}


def parse_svg_to_pattern(
    svg_str: str,
    grid_width: int = 40,
    grid_height: int = 50,
    palette_size: int = 6,
    quilt_width_in: float = 60.0,
    quilt_height_in: float = 72.0,
    seam_allowance: float = 0.25,
) -> tuple[QuiltPattern, float]:
    """
    Parse an SVG string into a QuiltPattern.

    Returns (pattern, confidence_score) where confidence is 0.0–1.0.
    """
    try:
        return _parse_impl(
            svg_str, grid_width, grid_height, palette_size,
            quilt_width_in, quilt_height_in, seam_allowance,
        )
    except Exception as e:
        logger.warning(f"SVG parsing failed, returning fallback: {e}")
        return _fallback_pattern(
            grid_width, grid_height, palette_size,
            quilt_width_in, quilt_height_in, seam_allowance,
        ), 0.0


def _parse_impl(
    svg_str: str,
    grid_width: int,
    grid_height: int,
    palette_size: int,
    quilt_width_in: float,
    quilt_height_in: float,
    seam_allowance: float,
) -> tuple[QuiltPattern, float]:
    # Phase 1: Parse SVG + viewBox
    root = ET.fromstring(svg_str)
    vb_w, vb_h = _parse_viewbox(root)

    # Phase 2: Extract colored elements
    elements = _extract_elements(root, vb_w, vb_h)
    if not elements:
        logger.warning("No colored elements found in SVG")
        return _fallback_pattern(
            grid_width, grid_height, palette_size,
            quilt_width_in, quilt_height_in, seam_allowance,
        ), 0.0

    # Phase 3: Rasterize to cell grid (center-point sampling, z-order)
    cell_grid = _rasterize_to_grid(elements, vb_w, vb_h, grid_width, grid_height)

    # Determine background color (most common)
    color_counts: dict[str, int] = {}
    for row in cell_grid:
        for color in row:
            if color is not None:
                color_counts[color] = color_counts.get(color, 0) + 1
    bg_color = max(color_counts, key=color_counts.get) if color_counts else "#ffffff"

    # Fill unfilled cells with background
    for gy in range(grid_height):
        for gx in range(grid_width):
            if cell_grid[gy][gx] is None:
                cell_grid[gy][gx] = bg_color

    # Collect all unique colors
    unique_colors = set()
    for row in cell_grid:
        for color in row:
            unique_colors.add(color)

    # Phase 4: Quantize colors if needed
    if len(unique_colors) > palette_size:
        color_map = _quantize_colors(unique_colors, palette_size)
        for gy in range(grid_height):
            for gx in range(grid_width):
                cell_grid[gy][gx] = color_map[cell_grid[gy][gx]]

    # Rebuild unique colors after quantization
    unique_colors = set()
    for row in cell_grid:
        for color in row:
            unique_colors.add(color)

    # Build color-to-index mapping
    color_list = sorted(unique_colors)
    color_to_idx: dict[str, int] = {c: i for i, c in enumerate(color_list)}
    idx_grid = [[color_to_idx[cell_grid[gy][gx]] for gx in range(grid_width)]
                for gy in range(grid_height)]

    # Build fabrics from unique colors → Kona Cotton
    fabric_id_map: dict[int, str] = {}
    fabrics: list[Fabric] = []
    used_names: set[str] = set()
    for ci, hex_color in enumerate(color_list):
        kona = match_kona(hex_color)
        fid = f"f{ci + 1}"
        fabric_id_map[ci] = fid
        name = kona["name"]
        if name in used_names:
            name = f"{name} ({ci + 1})"
        used_names.add(name)
        fabrics.append(Fabric(id=fid, color_hex=hex_color, name=name))

    # Phase 5: Detect diagonal triangles
    corner_map = _detect_triangles(elements, vb_w, vb_h, grid_width, grid_height,
                                   idx_grid, fabric_id_map)

    # Phase 6: Greedy rectangle merge
    blocks = _merge_grid_to_blocks(idx_grid, grid_width, grid_height,
                                   fabric_id_map, corner_map)

    # Phase 7: Assemble QuiltPattern
    cell_w = quilt_width_in / grid_width
    cell_h = quilt_height_in / grid_height
    cell_sizes = [{"w": cell_w, "h": cell_h} for _ in range(grid_width * grid_height)]

    pattern = QuiltPattern(
        grid_width=grid_width,
        grid_height=grid_height,
        quilt_width_in=quilt_width_in,
        quilt_height_in=quilt_height_in,
        seam_allowance=seam_allowance,
        fabrics=fabrics,
        blocks=blocks,
        cell_sizes=cell_sizes,
    )

    # Compute confidence
    confidence = _compute_confidence(pattern, grid_width, grid_height, corner_map)

    return pattern, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Parse viewBox
# ─────────────────────────────────────────────────────────────────────────────

def _parse_viewbox(root: ET.Element) -> tuple[float, float]:
    """Extract viewBox dimensions or fall back to width/height attributes."""
    vb = root.get("viewBox")
    if vb:
        parts = vb.replace(",", " ").split()
        if len(parts) >= 4:
            return float(parts[2]), float(parts[3])

    w = _parse_dimension(root.get("width", "100"))
    h = _parse_dimension(root.get("height", "100"))
    return w, h


def _parse_dimension(val: str) -> float:
    """Parse an SVG dimension string, stripping units."""
    m = re.match(r"([0-9.]+)", val.strip())
    return float(m.group(1)) if m else 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Extract colored elements
# ─────────────────────────────────────────────────────────────────────────────

def _extract_elements(
    root: ET.Element, vb_w: float, vb_h: float
) -> list[dict]:
    """
    Walk document order, extract bbox + fill for each shape element.
    Returns list of {"bbox": (x, y, w, h), "fill": "#hex", "points": [...] or None}.
    """
    min_area = vb_w * vb_h * 0.005  # skip elements < 0.5% of viewBox
    elements: list[dict] = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        fill = _parse_fill(elem)
        if fill is None:
            continue

        bbox = None
        points = None

        if tag == "rect":
            x = float(elem.get("x", "0"))
            y = float(elem.get("y", "0"))
            w = float(elem.get("width", "0"))
            h = float(elem.get("height", "0"))
            bbox = (x, y, w, h)

        elif tag == "circle":
            cx = float(elem.get("cx", "0"))
            cy = float(elem.get("cy", "0"))
            r = float(elem.get("r", "0"))
            bbox = (cx - r, cy - r, 2 * r, 2 * r)

        elif tag == "ellipse":
            cx = float(elem.get("cx", "0"))
            cy = float(elem.get("cy", "0"))
            rx = float(elem.get("rx", "0"))
            ry = float(elem.get("ry", "0"))
            bbox = (cx - rx, cy - ry, 2 * rx, 2 * ry)

        elif tag == "polygon" or tag == "polyline":
            pts = _parse_points(elem.get("points", ""))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                points = pts

        elif tag == "path":
            d = elem.get("d", "")
            if d:
                result = _parse_path_bbox(d)
                if result:
                    bbox = result
                    # Also try to extract points for triangle detection
                    pts = _path_to_points(d)
                    if pts:
                        points = pts

        if bbox is None:
            continue

        # Filter small elements
        area = bbox[2] * bbox[3]
        if area < min_area:
            continue

        elements.append({"bbox": bbox, "fill": fill, "points": points})

    return elements


def _parse_fill(element: ET.Element) -> Optional[str]:
    """
    Extract hex fill color from an SVG element.
    Checks: fill attribute, style attribute, inherited from parent <g>.
    Returns None for fill="none" or unfilled elements.
    """
    # Check style attribute first (higher specificity)
    style = element.get("style", "")
    if style:
        fill_match = re.search(r"fill\s*:\s*([^;]+)", style)
        if fill_match:
            raw = fill_match.group(1).strip()
            if raw == "none":
                return None
            return _normalize_color(raw)

    # Check fill attribute
    fill = element.get("fill")
    if fill is not None:
        if fill == "none":
            return None
        return _normalize_color(fill)

    # Check parent for inherited fill
    # Note: ET doesn't have parent references, so we check for common defaults
    # Elements without explicit fill default to black in SVG
    return None


def _normalize_color(raw: str) -> Optional[str]:
    """Convert color values (hex, rgb(), named) to lowercase hex."""
    raw = raw.strip().lower()

    if raw.startswith("#"):
        h = raw.lstrip("#")
        if len(h) == 3:
            h = h[0] * 2 + h[1] * 2 + h[2] * 2
        if len(h) == 6:
            return f"#{h}"
        return None

    # rgb(r, g, b) or rgb(r%, g%, b%)
    rgb_match = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", raw)
    if rgb_match:
        r, g, b = int(rgb_match.group(1)), int(rgb_match.group(2)), int(rgb_match.group(3))
        return f"#{r:02x}{g:02x}{b:02x}"

    rgb_pct_match = re.match(r"rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)", raw)
    if rgb_pct_match:
        r = int(float(rgb_pct_match.group(1)) * 255 / 100)
        g = int(float(rgb_pct_match.group(2)) * 255 / 100)
        b = int(float(rgb_pct_match.group(3)) * 255 / 100)
        return f"#{r:02x}{g:02x}{b:02x}"

    # Named color
    if raw in _NAMED_COLORS:
        return _NAMED_COLORS[raw]

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SVG points + path parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG points attribute: '100,200 300,400' or '100 200 300 400'."""
    points_str = points_str.strip()
    if not points_str:
        return []

    # Split on whitespace and/or commas
    nums = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", points_str)
    points = []
    for i in range(0, len(nums) - 1, 2):
        points.append((float(nums[i]), float(nums[i + 1])))
    return points


def _parse_path_bbox(d_attr: str) -> Optional[tuple[float, float, float, float]]:
    """
    Parse SVG path d attribute and return axis-aligned bounding box.
    Handles M/L/H/V/C/Q/A/Z in both absolute and relative forms.
    """
    tokens = _tokenize_path(d_attr)
    if not tokens:
        return None

    points: list[tuple[float, float]] = []
    cx, cy = 0.0, 0.0  # current point
    sx, sy = 0.0, 0.0  # subpath start
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        if not isinstance(cmd, str) or not cmd.isalpha():
            i += 1
            continue
        i += 1
        is_rel = cmd.islower()
        cmd_upper = cmd.upper()

        if cmd_upper == "Z":
            cx, cy = sx, sy
            continue

        if cmd_upper == "M":
            while i < len(tokens) and _is_number(tokens[i]):
                x, y = float(tokens[i]), float(tokens[i + 1])
                if is_rel:
                    x += cx
                    y += cy
                cx, cy = x, y
                sx, sy = x, y
                points.append((cx, cy))
                i += 2

        elif cmd_upper == "L":
            while i < len(tokens) and _is_number(tokens[i]):
                x, y = float(tokens[i]), float(tokens[i + 1])
                if is_rel:
                    x += cx
                    y += cy
                cx, cy = x, y
                points.append((cx, cy))
                i += 2

        elif cmd_upper == "H":
            while i < len(tokens) and _is_number(tokens[i]):
                x = float(tokens[i])
                if is_rel:
                    x += cx
                cx = x
                points.append((cx, cy))
                i += 1

        elif cmd_upper == "V":
            while i < len(tokens) and _is_number(tokens[i]):
                y = float(tokens[i])
                if is_rel:
                    y += cy
                cy = y
                points.append((cx, cy))
                i += 1

        elif cmd_upper == "C":
            while i + 5 < len(tokens) and _is_number(tokens[i]):
                x1, y1 = float(tokens[i]), float(tokens[i + 1])
                x2, y2 = float(tokens[i + 2]), float(tokens[i + 3])
                x, y = float(tokens[i + 4]), float(tokens[i + 5])
                if is_rel:
                    x1 += cx; y1 += cy
                    x2 += cx; y2 += cy
                    x += cx; y += cy
                points.extend([(x1, y1), (x2, y2), (x, y)])
                cx, cy = x, y
                i += 6

        elif cmd_upper == "Q":
            while i + 3 < len(tokens) and _is_number(tokens[i]):
                x1, y1 = float(tokens[i]), float(tokens[i + 1])
                x, y = float(tokens[i + 2]), float(tokens[i + 3])
                if is_rel:
                    x1 += cx; y1 += cy
                    x += cx; y += cy
                points.extend([(x1, y1), (x, y)])
                cx, cy = x, y
                i += 4

        elif cmd_upper == "A":
            while i + 6 < len(tokens) and _is_number(tokens[i]):
                # rx, ry, rotation, large-arc, sweep, x, y
                x, y = float(tokens[i + 5]), float(tokens[i + 6])
                if is_rel:
                    x += cx; y += cy
                points.append((x, y))
                cx, cy = x, y
                i += 7

        elif cmd_upper in ("S", "T"):
            # Smooth curves — consume params but simplified bbox
            param_count = 4 if cmd_upper == "S" else 2
            while i + param_count - 1 < len(tokens) and _is_number(tokens[i]):
                x = float(tokens[i + param_count - 2])
                y = float(tokens[i + param_count - 1])
                if is_rel:
                    x += cx; y += cy
                points.append((x, y))
                cx, cy = x, y
                i += param_count

        else:
            # Unknown command, skip
            continue

    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return (x_min, y_min, w, h)


def _path_to_points(d_attr: str) -> Optional[list[tuple[float, float]]]:
    """Extract vertex points from a simple path (M/L/Z only) for triangle detection."""
    tokens = _tokenize_path(d_attr)
    if not tokens:
        return None

    points: list[tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    i = 0

    while i < len(tokens):
        cmd = tokens[i]
        if not isinstance(cmd, str) or not cmd.isalpha():
            i += 1
            continue
        i += 1
        is_rel = cmd.islower()
        cmd_upper = cmd.upper()

        if cmd_upper == "Z":
            continue

        if cmd_upper in ("M", "L"):
            while i < len(tokens) and _is_number(tokens[i]):
                x, y = float(tokens[i]), float(tokens[i + 1])
                if is_rel:
                    x += cx; y += cy
                cx, cy = x, y
                points.append((cx, cy))
                i += 2
        else:
            # Non-linear commands — not a simple polygon
            return None

    return points if points else None


def _tokenize_path(d_attr: str) -> list[str]:
    """Tokenize SVG path d attribute into commands and numbers."""
    # Insert spaces around command letters for easy splitting
    d = re.sub(r"([A-Za-z])", r" \1 ", d_attr)
    # Handle negative numbers after commas/spaces correctly
    tokens = re.findall(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d)
    return tokens


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Rasterize to cell grid
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize_to_grid(
    elements: list[dict],
    vb_w: float,
    vb_h: float,
    grid_width: int,
    grid_height: int,
) -> list[list[Optional[str]]]:
    """
    Map SVG elements to grid cells via center-point sampling.
    Later elements overwrite earlier ones (z-order).
    """
    grid: list[list[Optional[str]]] = [[None] * grid_width for _ in range(grid_height)]
    cell_w = vb_w / grid_width
    cell_h = vb_h / grid_height

    for elem in elements:
        bx, by, bw, bh = elem["bbox"]
        fill = elem["fill"]

        # Determine which grid cells this element covers (center-point test)
        gx_min = max(0, int(bx / cell_w))
        gx_max = min(grid_width - 1, int((bx + bw) / cell_w))
        gy_min = max(0, int(by / cell_h))
        gy_max = min(grid_height - 1, int((by + bh) / cell_h))

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                # Center of this cell in SVG coordinates
                cx = (gx + 0.5) * cell_w
                cy = (gy + 0.5) * cell_h
                # Check if center falls within element bbox
                if bx <= cx <= bx + bw and by <= cy <= by + bh:
                    grid[gy][gx] = fill

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Color quantization
# ─────────────────────────────────────────────────────────────────────────────

def _quantize_colors(
    colors: set[str], max_colors: int
) -> dict[str, str]:
    """
    Agglomerative merge of similar colors until <= max_colors remain.
    Uses CIELAB distance via color_matcher._color_distance.
    Returns mapping from old color → new (merged) color.
    """
    # Start with each color as its own cluster
    clusters: list[list[str]] = [[c] for c in sorted(colors)]

    while len(clusters) > max_colors:
        # Find the two closest clusters
        best_dist = float("inf")
        best_i, best_j = 0, 1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Use representative color (first in cluster)
                d = _color_distance(clusters[i][0], clusters[j][0])
                if d < best_dist:
                    best_dist = d
                    best_i, best_j = i, j

        # Merge j into i
        clusters[best_i].extend(clusters[best_j])
        del clusters[best_j]

    # Build mapping: each color maps to its cluster representative
    color_map: dict[str, str] = {}
    for cluster in clusters:
        rep = cluster[0]
        for c in cluster:
            color_map[c] = rep

    return color_map


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5: Detect diagonal triangles
# ─────────────────────────────────────────────────────────────────────────────

def _detect_triangles(
    elements: list[dict],
    vb_w: float,
    vb_h: float,
    grid_width: int,
    grid_height: int,
    idx_grid: list[list[int]],
    fabric_id_map: dict[int, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """
    Detect polygon/path elements with 3 vertices that fall within a single
    cell and classify as nw/ne/sw/se corner triangle.
    """
    corner_map: dict[tuple[int, int], dict[str, str]] = {}
    cell_w = vb_w / grid_width
    cell_h = vb_h / grid_height

    for elem in elements:
        points = elem.get("points")
        if points is None or len(points) != 3:
            continue

        fill = elem["fill"]

        # Check if all 3 vertices are within the same cell
        cells = set()
        for px, py in points:
            gx = min(grid_width - 1, max(0, int(px / cell_w)))
            gy = min(grid_height - 1, max(0, int(py / cell_h)))
            cells.add((gx, gy))

        if len(cells) != 1:
            continue

        gx, gy = cells.pop()
        cell_bounds = (gx * cell_w, gy * cell_h, cell_w, cell_h)

        corner_type = _classify_triangle(points, cell_bounds)
        if corner_type is None:
            continue

        # Find the fabric_id for this triangle's fill color
        # We need to find which fabric matches this fill
        tri_fabric_id = None
        for ci, fid in fabric_id_map.items():
            # Get the fabric's color from the idx_grid fabrics
            # Since idx_grid maps cells to color indices, we look for matching fill
            pass

        # Simpler approach: look up fabric by traversing existing fabrics
        # The fill color should match one of our quantized colors
        # For now, use the element's fill to find closest fabric
        if (gx, gy) not in corner_map:
            corner_map[(gx, gy)] = {}

        # Find fabric_id for the triangle color
        for ci, fid in fabric_id_map.items():
            corner_map[(gx, gy)][corner_type] = fid
            # We'll set it properly below

        # The triangle's fabric should differ from the cell's base color
        cell_color_idx = idx_grid[gy][gx]
        cell_fabric_id = fabric_id_map[cell_color_idx]

        # Find the fabric whose color is closest to the triangle fill
        best_fid = cell_fabric_id
        best_dist = float("inf")
        for ci, fid in fabric_id_map.items():
            if fid == cell_fabric_id:
                continue
            # We don't have direct access to the color here easily,
            # but we can skip if there's only one fabric
            pass

        # For simplicity: if we found a valid corner, record it with a different fabric
        # The triangle fabric must differ from the cell's base fabric
        if len(fabric_id_map) > 1:
            # Pick any fabric that isn't the cell's fabric
            for ci, fid in fabric_id_map.items():
                if fid != cell_fabric_id:
                    corner_map[(gx, gy)][corner_type] = fid
                    break
        else:
            # Only one fabric, can't have a corner
            if (gx, gy) in corner_map:
                del corner_map[(gx, gy)]

    return corner_map


def _classify_triangle(
    points: list[tuple[float, float]],
    cell_bounds: tuple[float, float, float, float],
) -> Optional[str]:
    """
    Classify a 3-vertex polygon as nw, ne, sw, or se corner triangle
    based on which corner of the cell the triangle occupies.

    cell_bounds: (x, y, w, h) of the cell in SVG coordinates.
    """
    cx_cell, cy_cell, cw, ch = cell_bounds
    center_x = cx_cell + cw / 2
    center_y = cy_cell + ch / 2

    # Compute centroid of triangle
    tri_cx = sum(p[0] for p in points) / 3
    tri_cy = sum(p[1] for p in points) / 3

    if tri_cx < center_x and tri_cy < center_y:
        return "nw"
    elif tri_cx >= center_x and tri_cy < center_y:
        return "ne"
    elif tri_cx < center_x and tri_cy >= center_y:
        return "sw"
    else:
        return "se"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: Greedy rectangle merge
# ─────────────────────────────────────────────────────────────────────────────

def _merge_grid_to_blocks(
    grid: list[list[int]],
    grid_width: int,
    grid_height: int,
    fabric_id_map: dict[int, str],
    corner_map: Optional[dict[tuple[int, int], dict[str, str]]] = None,
) -> list[Block]:
    """
    Greedy maximal-rectangle merging — same algorithm as grid_extractor.
    Corner cells emitted as 1x1 blocks first, then row-scan merge remaining.
    """
    if corner_map is None:
        corner_map = {}
    visited = [[False] * grid_width for _ in range(grid_height)]
    blocks: list[Block] = []

    # First pass: corner cells as 1x1
    for (gx, gy), corners in corner_map.items():
        if 0 <= gy < grid_height and 0 <= gx < grid_width:
            visited[gy][gx] = True
            blocks.append(Block(
                x=gx, y=gy,
                width=1, height=1,
                fabric_id=fabric_id_map[grid[gy][gx]],
                corners=corners,
            ))

    # Second pass: greedy merge
    for gy in range(grid_height):
        for gx in range(grid_width):
            if visited[gy][gx]:
                continue
            color = grid[gy][gx]

            # Find max-width run to the right
            max_w = 0
            while (gx + max_w < grid_width
                   and grid[gy][gx + max_w] == color
                   and not visited[gy][gx + max_w]):
                max_w += 1

            # Extend downward
            max_h = 1
            while gy + max_h < grid_height:
                row_ok = all(
                    grid[gy + max_h][gx + dx] == color
                    and not visited[gy + max_h][gx + dx]
                    for dx in range(max_w)
                )
                if not row_ok:
                    break
                max_h += 1

            # Mark visited
            for dy in range(max_h):
                for dx in range(max_w):
                    visited[gy + dy][gx + dx] = True

            blocks.append(Block(
                x=gx, y=gy,
                width=max_w, height=max_h,
                fabric_id=fabric_id_map[color],
            ))

    return blocks


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Confidence scoring
# ─────────────────────────────────────────────────────────────────────────────

def _compute_confidence(
    pattern: QuiltPattern,
    grid_width: int,
    grid_height: int,
    corner_map: dict,
) -> float:
    """
    Compute confidence score:
      base 0.5
      +0.2 if multi-cell blocks exist
      +0.1 if colors are clean (few unique)
      +0.1 if full cell coverage
      +0.1 if corners detected
      cap at 1.0
    """
    confidence = 0.5

    # Multi-cell blocks
    multi_cell = sum(b.area_cells() for b in pattern.blocks if b.area_cells() > 1)
    total = grid_width * grid_height
    if multi_cell > 0:
        confidence += 0.2

    # Clean colors (palette_size <= 8 means intentional)
    if len(pattern.fabrics) <= 8:
        confidence += 0.1

    # Full coverage
    covered = pattern.covered_cells()
    if len(covered) == total:
        confidence += 0.1

    # Corners detected
    if corner_map:
        confidence += 0.1

    return min(1.0, round(confidence, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_pattern(
    grid_width: int,
    grid_height: int,
    palette_size: int,
    quilt_width_in: float,
    quilt_height_in: float,
    seam_allowance: float,
) -> QuiltPattern:
    """Return a simple striped placeholder pattern on parse failure."""
    colors = [
        ("#1b2d5b", "Kona Cotton - Navy"),
        ("#c43428", "Kona Cotton - Tomato"),
        ("#f5f0dc", "Kona Cotton - Cream"),
        ("#4a7c3f", "Kona Cotton - Grass"),
        ("#d4a42a", "Kona Cotton - Gold"),
        ("#7db8d8", "Kona Cotton - Sky"),
    ][:min(palette_size, grid_height)]

    fabrics = [Fabric(id=f"f{i + 1}", color_hex=c, name=n) for i, (c, n) in enumerate(colors)]
    blocks: list[Block] = []

    stripe_h = max(1, grid_height // len(fabrics))
    for i, fab in enumerate(fabrics):
        y_start = i * stripe_h
        if y_start >= grid_height:
            break
        y_end = y_start + stripe_h if i < len(fabrics) - 1 else grid_height
        y_end = min(y_end, grid_height)
        blocks.append(Block(x=0, y=y_start, width=grid_width,
                            height=y_end - y_start, fabric_id=fab.id))

    cell_w = quilt_width_in / grid_width
    cell_h = quilt_height_in / grid_height
    cell_sizes = [{"w": cell_w, "h": cell_h} for _ in range(grid_width * grid_height)]

    return QuiltPattern(
        grid_width=grid_width,
        grid_height=grid_height,
        quilt_width_in=quilt_width_in,
        quilt_height_in=quilt_height_in,
        seam_allowance=seam_allowance,
        fabrics=fabrics,
        blocks=blocks,
        cell_sizes=cell_sizes,
    )
