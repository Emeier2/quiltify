"""
Grid Extractor — turns an AI-generated quilt image into a structured QuiltPattern.

Pipeline:
  1. Resize image to (grid_width × cell_px, grid_height × cell_px)
  2. K-means color quantization to palette_size colors
  3. Sample color at each grid cell center
  4. Assign each cell its nearest quantized color
  5. Greedy rectangle merging: group adjacent same-color cells into Block objects
  6. Map quantized colors → Kona Cotton names via color_matcher
  7. Return QuiltPattern + confidence score (0.0–1.0)
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


CELL_SAMPLE_PX = 24   # pixels per grid cell during extraction


def extract_pattern_from_image(
    image_bytes: bytes,
    grid_width: int = 40,
    grid_height: int = 50,
    palette_size: int = 6,
    block_size_in: float = 2.5,
    seam_allowance: float = 0.25,
) -> tuple[QuiltPattern, float]:
    """
    Extract a QuiltPattern from raw image bytes.
    Returns (pattern, confidence_score) where confidence is 0.0–1.0.
    """
    if not _HAS_CV:
        return _synthetic_fallback(grid_width, grid_height, palette_size,
                                   block_size_in, seam_allowance), 0.0

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize to exact grid dimensions at our sample resolution
    target_w = grid_width * CELL_SAMPLE_PX
    target_h = grid_height * CELL_SAMPLE_PX
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

    # Greedy rectangle merging
    blocks = _merge_grid_to_blocks(grid, grid_width, grid_height, fabric_id_map)

    # Confidence score: ratio of cells covered by multi-cell blocks (vs 1×1 singletons)
    multi_cell = sum(b.area_cells() for b in blocks if b.area_cells() > 1)
    total_cells = grid_width * grid_height
    confidence = round(min(1.0, multi_cell / total_cells + 0.3), 3)

    pattern = QuiltPattern(
        grid_width=grid_width,
        grid_height=grid_height,
        block_size_in=block_size_in,
        seam_allowance=seam_allowance,
        fabrics=fabrics,
        blocks=blocks,
    )
    return pattern, confidence


def _merge_grid_to_blocks(
    grid: list[list[int]],
    grid_width: int,
    grid_height: int,
    fabric_id_map: dict[int, str],
) -> list[Block]:
    """
    Greedy maximal-rectangle merging:
    Scan row by row; for each unvisited cell, find the largest rectangle
    of the same color starting there.
    """
    visited = [[False] * grid_width for _ in range(grid_height)]
    blocks: list[Block] = []

    for gy in range(grid_height):
        for gx in range(grid_width):
            if visited[gy][gx]:
                continue
            color = grid[gy][gx]
            # Find max-width run to the right
            max_w = 0
            while gx + max_w < grid_width and grid[gy][gx + max_w] == color:
                max_w += 1
            # Extend downward while all cells in the row are same color and unvisited
            max_h = 1
            while gy + max_h < grid_height:
                row_ok = all(
                    grid[gy + max_h][gx + dx] == color and not visited[gy + max_h][gx + dx]
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
    block_size_in: float,
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

    return QuiltPattern(
        grid_width=grid_width,
        grid_height=grid_height,
        block_size_in=block_size_in,
        seam_allowance=seam_allowance,
        fabrics=fabrics,
        blocks=blocks,
    )
