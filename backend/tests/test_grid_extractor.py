"""Unit tests for grid_extractor.py — edge detection and corner classification."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import numpy as np

from backend.services.grid_extractor import (
    _detect_corners,
    _merge_grid_to_blocks,
    CELL_SAMPLE_PX,
)


def _make_solid_patch(color: list[int], sz: int = CELL_SAMPLE_PX) -> np.ndarray:
    """Create a solid-color cell patch."""
    patch = np.zeros((sz, sz, 3), dtype=np.uint8)
    patch[:, :] = color
    return patch


def _make_diagonal_patch(
    color_a: list[int],
    color_b: list[int],
    corner: str,
    sz: int = CELL_SAMPLE_PX,
) -> np.ndarray:
    """Create a cell patch split diagonally with two colors.

    corner specifies which triangle gets color_b:
      nw = upper-left triangle is color_b
      ne = upper-right triangle is color_b
      sw = lower-left triangle is color_b
      se = lower-right triangle is color_b
    """
    patch = np.zeros((sz, sz, 3), dtype=np.uint8)
    patch[:, :] = color_a
    for py in range(sz):
        for px in range(sz):
            if corner == "nw" and (py + px) < sz:
                patch[py, px] = color_b
            elif corner == "ne" and py < px:
                patch[py, px] = color_b
            elif corner == "sw" and py >= px:
                patch[py, px] = color_b
            elif corner == "se" and (py + px) >= sz:
                patch[py, px] = color_b
    return patch


def _build_image(patches: list[list[np.ndarray]]) -> np.ndarray:
    """Stitch cell patches into a full image array. patches[gy][gx]."""
    rows = []
    for row in patches:
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _detect_corners
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectCorners:
    def test_solid_cells_no_corners(self):
        """A uniform grid of one color should produce no corner detections."""
        red = [200, 50, 50]
        patches = [[_make_solid_patch(red) for _ in range(4)] for _ in range(4)]
        arr = _build_image(patches)
        grid = [[0] * 4 for _ in range(4)]
        centers = np.array([red], dtype=np.int32)
        fabric_id_map = {0: "f1"}

        corners = _detect_corners(arr, grid, 4, 4, centers, fabric_id_map)
        assert corners == {}

    def test_two_colors_no_diagonal_no_corners(self):
        """A sharp vertical boundary (left=red, right=blue) should not
        produce diagonal corners since the boundary is axis-aligned."""
        red = [200, 50, 50]
        blue = [50, 50, 200]
        patches = [
            [_make_solid_patch(red), _make_solid_patch(blue)],
            [_make_solid_patch(red), _make_solid_patch(blue)],
        ]
        arr = _build_image(patches)
        grid = [[0, 1], [0, 1]]
        centers = np.array([red, blue], dtype=np.int32)
        fabric_id_map = {0: "f1", 1: "f2"}

        corners = _detect_corners(arr, grid, 2, 2, centers, fabric_id_map)
        assert corners == {}

    def test_diagonal_nw_detected(self):
        """A cell with an NW diagonal split should detect an nw corner."""
        red = [200, 50, 50]
        blue = [50, 50, 200]
        # Single cell with diagonal: primary=red, nw triangle=blue
        diag_patch = _make_diagonal_patch(red, blue, "nw")
        arr = _build_image([[diag_patch]])
        # Cell is assigned to red (primary)
        grid = [[0]]
        centers = np.array([red, blue], dtype=np.int32)
        fabric_id_map = {0: "f1", 1: "f2"}

        corners = _detect_corners(arr, grid, 1, 1, centers, fabric_id_map)
        assert (0, 0) in corners
        assert "nw" in corners[(0, 0)]
        assert corners[(0, 0)]["nw"] == "f2"

    def test_diagonal_se_detected(self):
        """A cell with an SE diagonal split should detect an se corner."""
        red = [200, 50, 50]
        blue = [50, 50, 200]
        diag_patch = _make_diagonal_patch(red, blue, "se")
        arr = _build_image([[diag_patch]])
        grid = [[0]]
        centers = np.array([red, blue], dtype=np.int32)
        fabric_id_map = {0: "f1", 1: "f2"}

        corners = _detect_corners(arr, grid, 1, 1, centers, fabric_id_map)
        assert (0, 0) in corners
        assert "se" in corners[(0, 0)]
        assert corners[(0, 0)]["se"] == "f2"

    def test_more_than_two_corners_rejected(self):
        """If most of a cell is a different color (3-4 quadrants), it should
        NOT be classified as a corner — the cell should just be a different
        primary color, not a diagonal transition."""
        red = [200, 50, 50]
        blue = [50, 50, 200]
        # Make a patch that's ~90% blue — primary assignment is wrong (red)
        # but the detector should reject it (too many corners = not diagonal)
        patch = np.full((CELL_SAMPLE_PX, CELL_SAMPLE_PX, 3), blue, dtype=np.uint8)
        # Just a tiny red corner
        patch[0:3, 0:3] = red
        arr = _build_image([[patch]])
        grid = [[0]]  # wrongly assigned to red
        centers = np.array([red, blue], dtype=np.int32)
        fabric_id_map = {0: "f1", 1: "f2"}

        corners = _detect_corners(arr, grid, 1, 1, centers, fabric_id_map)
        # Should be rejected — 3-4 quadrants are blue, not a diagonal
        assert (0, 0) not in corners


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _merge_grid_to_blocks with corner_map
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeWithCorners:
    def test_no_corners_merges_normally(self):
        """Without corners, merge should work like before."""
        grid = [[0, 0, 1], [0, 0, 1]]
        fabric_id_map = {0: "f1", 1: "f2"}
        blocks = _merge_grid_to_blocks(grid, 3, 2, fabric_id_map)
        # Should merge: one 2x2 f1 block, one 1x2 f2 block
        assert len(blocks) == 2
        areas = sorted([b.width * b.height for b in blocks])
        assert areas == [2, 4]

    def test_corner_cell_stays_1x1(self):
        """A cell in corner_map should not be merged into larger blocks."""
        grid = [[0, 0, 0], [0, 0, 0]]
        fabric_id_map = {0: "f1", 1: "f2"}
        corner_map = {(1, 0): {"ne": "f2"}}

        blocks = _merge_grid_to_blocks(grid, 3, 2, fabric_id_map, corner_map)

        # Find the corner block
        corner_blocks = [b for b in blocks if b.corners]
        assert len(corner_blocks) == 1
        cb = corner_blocks[0]
        assert cb.x == 1 and cb.y == 0
        assert cb.width == 1 and cb.height == 1
        assert cb.corners == {"ne": "f2"}

    def test_corner_does_not_block_adjacent_merge(self):
        """Solid cells adjacent to a corner cell should still merge with
        each other (just not with the corner cell)."""
        # 4x1 row: all same color, cell 1 has a corner
        grid = [[0, 0, 0, 0]]
        fabric_id_map = {0: "f1", 1: "f2"}
        corner_map = {(1, 0): {"sw": "f2"}}

        blocks = _merge_grid_to_blocks(grid, 4, 1, fabric_id_map, corner_map)

        # Expect: (0,0) 1x1 solid, (1,0) 1x1 corner, (2,0) 2x1 solid
        assert len(blocks) == 3
        corner_blocks = [b for b in blocks if b.corners]
        assert len(corner_blocks) == 1
        solid_blocks = [b for b in blocks if not b.corners]
        assert len(solid_blocks) == 2

    def test_all_cells_covered(self):
        """Every cell should be covered exactly once with corners present."""
        grid = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
        fabric_id_map = {0: "f1", 1: "f2"}
        corner_map = {(0, 0): {"se": "f2"}, (2, 2): {"nw": "f2"}}

        blocks = _merge_grid_to_blocks(grid, 3, 3, fabric_id_map, corner_map)

        # Check coverage
        covered = set()
        for b in blocks:
            for dy in range(b.height):
                for dx in range(b.width):
                    cell = (b.x + dx, b.y + dy)
                    assert cell not in covered, f"Overlap at {cell}"
                    covered.add(cell)
        expected = {(x, y) for y in range(3) for x in range(3)}
        assert covered == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
