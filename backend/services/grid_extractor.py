"""
Grid Extractor — turns an AI-generated quilt image into a structured QuiltPattern.

Pipeline:
  1. Resize image to (grid_width × cell_px, grid_height × cell_px)
  2. K-means color quantization to palette_size colors
  3. Sample color at each grid cell center
  4. Assign each cell its nearest quantized color
  5. Detect diagonal color boundaries per cell → stitch-and-flip corners
  6. Greedy rectangle merging (skip corner cells) → Block objects
  7. Map quantized colors → Kona Cotton names via color_matcher
  8. Return QuiltPattern + confidence score (0.0–1.0)
"""
from __future__ import annotations

import base64
import hashlib
import io
import math
from typing import Optional

try:
    import numpy as np
    from PIL import Image
    from sklearn.cluster import KMeans, MiniBatchKMeans
    _HAS_CV = True
except ImportError:
    _HAS_CV = False

from .grid_engine import QuiltPattern, Block, Fabric
from .color_matcher import match_kona
from . import vtracer_service


CELL_SAMPLE_PX = 24   # pixels per grid cell during extraction

# Minimum fraction of a cell's diagonal that must show a color boundary
# for it to be classified as a corner cell.
_CORNER_EDGE_THRESHOLD = 0.35


def extract_pattern_from_image(
    image_bytes: bytes,
    grid_width: int = 40,
    grid_height: int = 50,
    palette_size: int = 6,
    quilt_width_in: float = 60.0,
    quilt_height_in: float = 72.0,
    seam_allowance: float = 0.25,
) -> tuple[QuiltPattern, float]:
    """
    Extract a QuiltPattern from raw image bytes.
    Returns (pattern, confidence_score) where confidence is 0.0–1.0.
    """
    if not _HAS_CV:
        return _synthetic_fallback(grid_width, grid_height, palette_size,
                                   quilt_width_in, quilt_height_in, seam_allowance), 0.0

    # Optional: vectorize then re-rasterize for cleaner color boundaries
    cleaned = vtracer_service.clean_for_extraction(
        image_bytes, grid_width, grid_height, CELL_SAMPLE_PX
    )
    if cleaned is not None:
        img = Image.open(io.BytesIO(cleaned)).convert("RGB")
    else:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to exact grid dimensions at our sample resolution
    target_w = grid_width * CELL_SAMPLE_PX
    target_h = grid_height * CELL_SAMPLE_PX
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)

    pixels = np.array(img).reshape(-1, 3).astype(np.float32)

    # K-means quantization
    n_clusters = min(palette_size, len(np.unique(pixels.reshape(-1, 3), axis=0)))
    n_clusters = max(2, n_clusters)

    km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)  # shape: (n_clusters, 3)

    # Sample each cell center
    cell_colors: list[list[int]] = []  # flat list, row-major
    half = CELL_SAMPLE_PX // 2
    arr = np.array(img)

    for gy in range(grid_height):
        for gx in range(grid_width):
            cx = gx * CELL_SAMPLE_PX + half
            cy = gy * CELL_SAMPLE_PX + half
            # Sample a 3×3 block at center and take median
            patch = arr[cy-1:cy+2, cx-1:cx+2].reshape(-1, 3)
            median_color = np.median(patch, axis=0).astype(int)
            cell_colors.append(median_color.tolist())

    # Assign each cell to nearest quantized color
    def nearest_center(rgb: list[int]) -> int:
        dists = [math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, c))) for c in centers.tolist()]
        return int(min(range(len(dists)), key=lambda i: dists[i]))

    cell_assignments: list[int] = [nearest_center(c) for c in cell_colors]
    # Shape: (grid_height, grid_width)
    grid = [[cell_assignments[gy * grid_width + gx] for gx in range(grid_width)]
            for gy in range(grid_height)]

    # Build fabrics from cluster centers
    fabric_id_map: dict[int, str] = {}
    fabrics: list[Fabric] = []
    for ci, center in enumerate(centers.tolist()):
        hex_color = "#{:02x}{:02x}{:02x}".format(int(center[0]), int(center[1]), int(center[2]))
        kona = match_kona(hex_color)
        fid = f"f{ci+1}"
        fabric_id_map[ci] = fid
        # Use unique kona name (append suffix if duplicate)
        name = kona["name"]
        existing_names = {f.name for f in fabrics}
        if name in existing_names:
            name = f"{name} ({ci+1})"
        fabrics.append(Fabric(id=fid, color_hex=hex_color, name=name))

    # Detect diagonal color boundaries → corner cells
    corner_map = _detect_corners(arr, grid, grid_width, grid_height, centers, fabric_id_map)

    # Greedy rectangle merging (skip corner cells)
    blocks = _merge_grid_to_blocks(grid, grid_width, grid_height, fabric_id_map, corner_map)

    # Confidence score: ratio of cells covered by multi-cell blocks (vs 1×1 singletons)
    multi_cell = sum(b.area_cells() for b in blocks if b.area_cells() > 1)
    total_cells = grid_width * grid_height
    confidence = round(min(1.0, multi_cell / total_cells + 0.3), 3)

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
    return pattern, confidence


def _detect_corners(
    arr: "np.ndarray",
    grid: list[list[int]],
    grid_width: int,
    grid_height: int,
    centers: "np.ndarray",
    fabric_id_map: dict[int, str],
) -> dict[tuple[int, int], dict[str, str]]:
    """
    For each cell, check if a diagonal color boundary crosses it by
    comparing the two halves along each possible diagonal.

    Diagonal 1 (TR-BL line, xs + ys = sz): splits cell into NW and SE halves.
    Diagonal 2 (TL-BR line, ys = xs): splits cell into NE and SW halves.

    If the two halves of a diagonal have different dominant quantized colors,
    the non-primary half is classified as a stitch-and-flip corner.

    Returns a dict mapping (gx, gy) -> {"nw": fabric_id, ...} for cells
    that should become corner blocks.
    """
    corner_map: dict[tuple[int, int], dict[str, str]] = {}
    sz = CELL_SAMPLE_PX
    ys, xs = np.mgrid[0:sz, 0:sz]
    # Diagonal 1 masks: TR-BL line (xs + ys = sz)
    mask_nw = (ys + xs) < sz       # upper-left half
    mask_se = ~mask_nw             # lower-right half
    # Diagonal 2 masks: TL-BR line (ys = xs)
    mask_ne = ys < xs              # upper-right half
    mask_sw = ~mask_ne             # lower-left half

    centers_f = centers.astype(np.float32)

    for gy in range(grid_height):
        for gx in range(grid_width):
            x0 = gx * sz
            y0 = gy * sz
            patch = arr[y0:y0 + sz, x0:x0 + sz]  # (sz, sz, 3)
            if patch.shape[0] < sz or patch.shape[1] < sz:
                continue

            primary = grid[gy][gx]
            corners: dict[str, str] = {}

            # Check diagonal 1: NW vs SE
            _check_diagonal(
                patch, mask_nw, mask_se, "nw", "se",
                primary, centers_f, fabric_id_map, corners,
            )
            # Check diagonal 2: NE vs SW
            _check_diagonal(
                patch, mask_ne, mask_sw, "ne", "sw",
                primary, centers_f, fabric_id_map, corners,
            )

            if corners:
                corner_map[(gx, gy)] = corners

    return corner_map


def _check_diagonal(
    patch: "np.ndarray",
    mask_a: "np.ndarray",
    mask_b: "np.ndarray",
    name_a: str,
    name_b: str,
    primary: int,
    centers_f: "np.ndarray",
    fabric_id_map: dict[int, str],
    corners: dict[str, str],
) -> None:
    """Check one diagonal split for a color boundary.

    If the two halves have different dominant colors AND the non-primary
    half meets the threshold, record it as a corner.
    """
    pixels_a = patch[mask_a].reshape(-1, 3).astype(np.float32)
    pixels_b = patch[mask_b].reshape(-1, 3).astype(np.float32)
    if len(pixels_a) == 0 or len(pixels_b) == 0:
        return

    color_a = int(np.sqrt(((centers_f - pixels_a.mean(axis=0)) ** 2).sum(axis=1)).argmin())
    color_b = int(np.sqrt(((centers_f - pixels_b.mean(axis=0)) ** 2).sum(axis=1)).argmin())

    # Both halves same color → no diagonal boundary
    if color_a == color_b:
        return

    # Determine which half is the secondary (non-primary) color
    if color_a != primary:
        secondary = color_a
        sec_pixels = pixels_a
        corner_name = name_a
    elif color_b != primary:
        secondary = color_b
        sec_pixels = pixels_b
        corner_name = name_b
    else:
        return

    # Verify the secondary half is strongly dominated by that color
    d_pri = np.sqrt(((sec_pixels - centers_f[primary]) ** 2).sum(axis=1))
    d_sec = np.sqrt(((sec_pixels - centers_f[secondary]) ** 2).sum(axis=1))
    frac = (d_sec < d_pri).sum() / len(sec_pixels)

    if frac >= _CORNER_EDGE_THRESHOLD:
        corners[corner_name] = fabric_id_map[secondary]


def _merge_grid_to_blocks(
    grid: list[list[int]],
    grid_width: int,
    grid_height: int,
    fabric_id_map: dict[int, str],
    corner_map: Optional[dict[tuple[int, int], dict[str, str]]] = None,
) -> list[Block]:
    """
    Greedy maximal-rectangle merging:
    Scan row by row; for each unvisited cell, find the largest rectangle
    of the same color starting there.  Corner cells (from edge detection)
    are emitted as 1×1 blocks and excluded from merging.
    """
    if corner_map is None:
        corner_map = {}
    visited = [[False] * grid_width for _ in range(grid_height)]
    blocks: list[Block] = []

    # First pass: emit corner cells as 1×1 blocks
    for (gx, gy), corners in corner_map.items():
        visited[gy][gx] = True
        blocks.append(Block(
            x=gx, y=gy,
            width=1, height=1,
            fabric_id=fabric_id_map[grid[gy][gx]],
            corners=corners,
        ))

    # Second pass: greedy merge remaining solid cells
    for gy in range(grid_height):
        for gx in range(grid_width):
            if visited[gy][gx]:
                continue
            color = grid[gy][gx]
            # Find max-width run to the right (skip corner cells)
            max_w = 0
            while (gx + max_w < grid_width
                   and grid[gy][gx + max_w] == color
                   and not visited[gy][gx + max_w]):
                max_w += 1
            # Extend downward while all cells in the row are same color and unvisited
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


def image_bytes_from_base64(b64: str) -> bytes:
    """Strip data URI prefix if present and decode base64."""
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _synthetic_fallback(
    grid_width: int,
    grid_height: int,
    palette_size: int,
    quilt_width_in: float,
    quilt_height_in: float,
    seam_allowance: float,
) -> QuiltPattern:
    """Return a striped placeholder pattern when image libraries are unavailable."""
    colors = [
        ("#1b2d5b", "Kona Cotton - Navy"),
        ("#c43428", "Kona Cotton - Tomato"),
        ("#f5f0dc", "Kona Cotton - Cream"),
        ("#4a7c3f", "Kona Cotton - Grass"),
        ("#d4a42a", "Kona Cotton - Gold"),
        ("#7db8d8", "Kona Cotton - Sky"),
    ][:palette_size]

    fabrics = [Fabric(id=f"f{i+1}", color_hex=c, name=n) for i, (c, n) in enumerate(colors)]
    blocks: list[Block] = []

    stripe_height = max(1, grid_height // palette_size)
    for i, fab in enumerate(fabrics):
        y_start = i * stripe_height
        y_end = y_start + stripe_height if i < len(fabrics) - 1 else grid_height
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
