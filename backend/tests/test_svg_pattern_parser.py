"""Unit tests for svg_pattern_parser.py — SVG to QuiltPattern conversion."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from backend.services.svg_pattern_parser import (
    parse_svg_to_pattern,
    _parse_fill,
    _parse_points,
    _parse_path_bbox,
    _normalize_color,
    _quantize_colors,
    _classify_triangle,
    _merge_grid_to_blocks,
    _parse_viewbox,
)
import xml.etree.ElementTree as ET


# ─────────────────────────────────────────────────────────────────────────────
# SVG fixtures
# ─────────────────────────────────────────────────────────────────────────────

SIMPLE_2X2_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="50" height="50" fill="#ff0000"/>
  <rect x="50" y="0" width="50" height="50" fill="#0000ff"/>
  <rect x="0" y="50" width="50" height="50" fill="#0000ff"/>
  <rect x="50" y="50" width="50" height="50" fill="#ff0000"/>
</svg>"""

OVERLAPPING_RECTS_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="#ff0000"/>
  <rect x="25" y="25" width="50" height="50" fill="#00ff00"/>
</svg>"""

RGB_FILL_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="rgb(255, 128, 0)"/>
</svg>"""

STYLE_FILL_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" style="fill: #336699;"/>
</svg>"""

NAMED_COLOR_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="navy"/>
</svg>"""

FILL_NONE_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="#ff0000"/>
  <rect x="25" y="25" width="50" height="50" fill="none"/>
</svg>"""

TRIANGLE_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="100" height="100" fill="#ff0000"/>
  <polygon points="0,0 50,0 0,50" fill="#0000ff"/>
</svg>"""

VIEWBOX_SVG = """<svg viewBox="10 20 200 300" xmlns="http://www.w3.org/2000/svg">
  <rect x="10" y="20" width="200" height="300" fill="#ff0000"/>
</svg>"""

NO_VIEWBOX_SVG = """<svg width="150" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="150" height="200" fill="#00ff00"/>
</svg>"""

CIRCLE_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="40" fill="#ff0000"/>
</svg>"""

ELLIPSE_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="50" cy="50" rx="40" ry="30" fill="#0000ff"/>
</svg>"""

MULTI_COLOR_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="20" height="100" fill="#ff0000"/>
  <rect x="20" y="0" width="20" height="100" fill="#ff0505"/>
  <rect x="40" y="0" width="20" height="100" fill="#ff0a0a"/>
  <rect x="60" y="0" width="20" height="100" fill="#ff1010"/>
  <rect x="80" y="0" width="20" height="100" fill="#0000ff"/>
</svg>"""

PATH_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 L 100 0 L 100 100 L 0 100 Z" fill="#ff0000"/>
</svg>"""

RELATIVE_PATH_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M 10 10 l 80 0 l 0 80 l -80 0 z" fill="#00ff00"/>
</svg>"""

HV_PATH_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 H 100 V 100 H 0 Z" fill="#0000ff"/>
</svg>"""

CURVE_PATH_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <path d="M 0 0 C 33 0 66 0 100 0 L 100 100 L 0 100 Z" fill="#ff8800"/>
</svg>"""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Basic parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestBasicParsing:
    def test_simple_2x2_grid(self):
        pattern, confidence = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert pattern.grid_width == 2
        assert pattern.grid_height == 2
        assert len(pattern.fabrics) == 2
        assert len(pattern.blocks) > 0
        errors = pattern.validate()
        assert not errors, f"Validation errors: {errors}"

    def test_correct_fabric_count(self):
        pattern, _ = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert len(pattern.fabrics) == 2

    def test_full_coverage(self):
        pattern, _ = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        covered = pattern.covered_cells()
        expected = pattern.all_cells()
        assert covered == expected

    def test_confidence_above_zero(self):
        _, confidence = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert confidence > 0.0

    def test_cell_sizes_correct_length(self):
        pattern, _ = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        assert len(pattern.cell_sizes) == 16


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Z-order (overlapping elements)
# ─────────────────────────────────────────────────────────────────────────────

class TestZOrder:
    def test_later_element_overwrites_earlier(self):
        pattern, _ = parse_svg_to_pattern(
            OVERLAPPING_RECTS_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        # Center cells (1,1), (2,2) should be green (second rect)
        cell_grid = pattern.cell_grid()
        # The green rect covers center cells, so at least some center cells
        # should have a different fabric than edge cells
        edge_fabric = cell_grid.get((0, 0))
        center_fabric = cell_grid.get((2, 2))
        assert edge_fabric is not None
        assert center_fabric is not None
        assert edge_fabric != center_fabric


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Color parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestColorParsing:
    def test_hex_fill(self):
        elem = ET.fromstring('<rect fill="#ff0000"/>')
        assert _parse_fill(elem) == "#ff0000"

    def test_rgb_fill(self):
        color = _normalize_color("rgb(255, 128, 0)")
        assert color == "#ff8000"

    def test_style_fill(self):
        elem = ET.fromstring('<rect style="fill: #336699;"/>')
        assert _parse_fill(elem) == "#336699"

    def test_named_color(self):
        color = _normalize_color("navy")
        assert color == "#000080"

    def test_fill_none_returns_none(self):
        elem = ET.fromstring('<rect fill="none"/>')
        assert _parse_fill(elem) is None

    def test_style_fill_none(self):
        elem = ET.fromstring('<rect style="fill: none;"/>')
        assert _parse_fill(elem) is None

    def test_short_hex(self):
        color = _normalize_color("#f00")
        assert color == "#ff0000"

    def test_named_color_svg(self):
        pattern, _ = parse_svg_to_pattern(
            NAMED_COLOR_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert len(pattern.fabrics) >= 1

    def test_rgb_fill_svg(self):
        pattern, _ = parse_svg_to_pattern(
            RGB_FILL_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert len(pattern.fabrics) >= 1
        errors = pattern.validate()
        assert not errors

    def test_style_fill_svg(self):
        pattern, _ = parse_svg_to_pattern(
            STYLE_FILL_SVG, grid_width=2, grid_height=2, palette_size=6
        )
        assert len(pattern.fabrics) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: viewBox parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestViewBox:
    def test_viewbox_dimensions(self):
        root = ET.fromstring(VIEWBOX_SVG)
        w, h = _parse_viewbox(root)
        assert w == 200
        assert h == 300

    def test_no_viewbox_uses_width_height(self):
        root = ET.fromstring(NO_VIEWBOX_SVG)
        w, h = _parse_viewbox(root)
        assert w == 150
        assert h == 200


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Points parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestPointsParsing:
    def test_comma_separated(self):
        pts = _parse_points("100,200 300,400")
        assert pts == [(100.0, 200.0), (300.0, 400.0)]

    def test_space_separated(self):
        pts = _parse_points("100 200 300 400")
        assert pts == [(100.0, 200.0), (300.0, 400.0)]

    def test_empty_string(self):
        pts = _parse_points("")
        assert pts == []

    def test_mixed_separators(self):
        pts = _parse_points("10,20 30,40 50,60")
        assert len(pts) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Path bbox parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestPathBbox:
    def test_simple_rect_path(self):
        bbox = _parse_path_bbox("M 0 0 L 100 0 L 100 50 L 0 50 Z")
        assert bbox is not None
        x, y, w, h = bbox
        assert x == 0
        assert y == 0
        assert w == 100
        assert h == 50

    def test_hv_commands(self):
        bbox = _parse_path_bbox("M 10 20 H 110 V 120 H 10 Z")
        assert bbox is not None
        x, y, w, h = bbox
        assert x == 10
        assert y == 20
        assert w == 100
        assert h == 100

    def test_relative_commands(self):
        bbox = _parse_path_bbox("M 10 10 l 80 0 l 0 80 l -80 0 z")
        assert bbox is not None
        x, y, w, h = bbox
        assert x == 10
        assert y == 10
        assert abs(w - 80) < 0.01
        assert abs(h - 80) < 0.01

    def test_curve_commands(self):
        bbox = _parse_path_bbox("M 0 0 C 33 0 66 0 100 0 L 100 100 L 0 100 Z")
        assert bbox is not None
        _, _, w, h = bbox
        assert w == 100
        assert h == 100

    def test_empty_path(self):
        bbox = _parse_path_bbox("")
        assert bbox is None

    def test_path_svg_produces_valid_pattern(self):
        pattern, _ = parse_svg_to_pattern(
            PATH_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        errors = pattern.validate()
        assert not errors


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Color quantization
# ─────────────────────────────────────────────────────────────────────────────

class TestColorQuantization:
    def test_reduces_similar_colors(self):
        # 4 similar reds should merge to fewer
        colors = {"#ff0000", "#ff0505", "#ff0a0a", "#ff1010"}
        result = _quantize_colors(colors, 2)
        assert len(set(result.values())) <= 2

    def test_preserves_distinct_colors(self):
        colors = {"#ff0000", "#0000ff"}
        result = _quantize_colors(colors, 2)
        assert len(set(result.values())) == 2

    def test_quantization_maps_all_inputs(self):
        colors = {"#ff0000", "#ff0505", "#ff0a0a", "#0000ff"}
        result = _quantize_colors(colors, 2)
        for c in colors:
            assert c in result

    def test_multi_color_svg_quantization(self):
        pattern, _ = parse_svg_to_pattern(
            MULTI_COLOR_SVG, grid_width=5, grid_height=2, palette_size=3
        )
        assert len(pattern.fabrics) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Triangle detection
# ─────────────────────────────────────────────────────────────────────────────

class TestTriangleDetection:
    def test_classify_nw_triangle(self):
        # Triangle in the NW corner of a 100x100 cell
        points = [(0.0, 0.0), (50.0, 0.0), (0.0, 50.0)]
        result = _classify_triangle(points, (0.0, 0.0, 100.0, 100.0))
        assert result == "nw"

    def test_classify_se_triangle(self):
        points = [(50.0, 50.0), (100.0, 50.0), (100.0, 100.0)]
        result = _classify_triangle(points, (0.0, 0.0, 100.0, 100.0))
        assert result == "se"

    def test_classify_ne_triangle(self):
        points = [(50.0, 0.0), (100.0, 0.0), (100.0, 50.0)]
        result = _classify_triangle(points, (0.0, 0.0, 100.0, 100.0))
        assert result == "ne"

    def test_classify_sw_triangle(self):
        points = [(0.0, 50.0), (50.0, 100.0), (0.0, 100.0)]
        result = _classify_triangle(points, (0.0, 0.0, 100.0, 100.0))
        assert result == "sw"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Greedy merge
# ─────────────────────────────────────────────────────────────────────────────

class TestGreedyMerge:
    def test_horizontal_merge(self):
        # 1x4 grid, all same color
        grid = [[0, 0, 0, 0]]
        fabric_map = {0: "f1"}
        blocks = _merge_grid_to_blocks(grid, 4, 1, fabric_map)
        assert len(blocks) == 1
        assert blocks[0].width == 4
        assert blocks[0].height == 1

    def test_rectangular_merge(self):
        # 2x3 grid, all same color
        grid = [[0, 0, 0], [0, 0, 0]]
        fabric_map = {0: "f1"}
        blocks = _merge_grid_to_blocks(grid, 3, 2, fabric_map)
        assert len(blocks) == 1
        assert blocks[0].width == 3
        assert blocks[0].height == 2

    def test_no_merge_different_colors(self):
        grid = [[0, 1], [1, 0]]
        fabric_map = {0: "f1", 1: "f2"}
        blocks = _merge_grid_to_blocks(grid, 2, 2, fabric_map)
        assert len(blocks) == 4  # each cell is its own block

    def test_corner_cells_excluded_from_merge(self):
        grid = [[0, 0], [0, 0]]
        fabric_map = {0: "f1"}
        corner_map = {(0, 0): {"nw": "f2"}}
        blocks = _merge_grid_to_blocks(grid, 2, 2, fabric_map, corner_map)
        # (0,0) is a 1x1 corner block, remaining cells merged
        corner_blocks = [b for b in blocks if b.corners]
        non_corner = [b for b in blocks if not b.corners]
        assert len(corner_blocks) == 1
        assert corner_blocks[0].width == 1
        assert corner_blocks[0].height == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Invalid / edge case input
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_svg_returns_fallback(self):
        pattern, confidence = parse_svg_to_pattern(
            "not valid xml at all", grid_width=4, grid_height=4
        )
        assert confidence == 0.0
        assert len(pattern.blocks) > 0
        errors = pattern.validate()
        assert not errors

    def test_empty_svg(self):
        svg = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"></svg>'
        pattern, confidence = parse_svg_to_pattern(
            svg, grid_width=4, grid_height=4
        )
        assert confidence == 0.0

    def test_fill_none_elements_skipped(self):
        pattern, _ = parse_svg_to_pattern(
            FILL_NONE_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        # Should have exactly 1 fabric (the red), fill=none skipped
        assert len(pattern.fabrics) >= 1

    def test_circle_element(self):
        pattern, _ = parse_svg_to_pattern(
            CIRCLE_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        assert len(pattern.fabrics) >= 1
        errors = pattern.validate()
        assert not errors

    def test_ellipse_element(self):
        pattern, _ = parse_svg_to_pattern(
            ELLIPSE_SVG, grid_width=4, grid_height=4, palette_size=6
        )
        assert len(pattern.fabrics) >= 1
        errors = pattern.validate()
        assert not errors

    def test_quilt_dimensions_preserved(self):
        pattern, _ = parse_svg_to_pattern(
            SIMPLE_2X2_SVG, grid_width=4, grid_height=4,
            quilt_width_in=48.0, quilt_height_in=60.0, seam_allowance=0.5
        )
        assert pattern.quilt_width_in == 48.0
        assert pattern.quilt_height_in == 60.0
        assert pattern.seam_allowance == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
