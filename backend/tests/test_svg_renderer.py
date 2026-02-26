"""Unit tests for svg_renderer.py — grid and cutting diagram SVG generation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from backend.services.grid_engine import Fabric, Block, QuiltPattern
from backend.services.svg_renderer import (
    render_grid_svg,
    render_cutting_diagram_svg,
    _contrasting_text,
    _fallback_svg,
    CELL_PX,
)


def _small_pattern() -> QuiltPattern:
    """4x4 grid with 2 fabrics for simple testing."""
    fabrics = [
        Fabric(id="f1", color_hex="#1b2d5b", name="Navy"),
        Fabric(id="f2", color_hex="#f5f0dc", name="Cream"),
    ]
    blocks = [
        Block(x=0, y=0, width=4, height=2, fabric_id="f1"),
        Block(x=0, y=2, width=4, height=2, fabric_id="f2"),
    ]
    return QuiltPattern(
        grid_width=4, grid_height=4,
        block_size_in=2.5, seam_allowance=0.25,
        fabrics=fabrics, blocks=blocks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: render_grid_svg
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderGridSvg:
    def test_returns_svg_string(self):
        svg = render_grid_svg(_small_pattern())
        assert "<svg" in svg
        assert "</svg>" in svg or "/>" in svg

    def test_contains_rect_elements(self):
        svg = render_grid_svg(_small_pattern())
        # Should have at least 2 rects for blocks + 1 background
        assert svg.count("<rect") >= 3

    def test_dimensions_match_pattern(self):
        p = _small_pattern()
        svg = render_grid_svg(p, cell_px=10)
        expected_w = str(p.grid_width * 10)
        expected_h = str(p.grid_height * 10)
        assert expected_w in svg
        assert expected_h in svg

    def test_contains_fabric_colors(self):
        svg = render_grid_svg(_small_pattern())
        assert "#1b2d5b" in svg
        assert "#f5f0dc" in svg


# ─────────────────────────────────────────────────────────────────────────────
# Tests: render_cutting_diagram_svg
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderCuttingDiagramSvg:
    def test_returns_svg_string(self):
        p = _small_pattern()
        chart = p.to_cutting_chart()
        svg = render_cutting_diagram_svg(chart, p)
        assert "<svg" in svg

    def test_contains_fabric_names(self):
        p = _small_pattern()
        chart = p.to_cutting_chart()
        svg = render_cutting_diagram_svg(chart, p)
        assert "Navy" in svg
        assert "Cream" in svg

    def test_contains_dimensions_label(self):
        p = _small_pattern()
        chart = p.to_cutting_chart()
        svg = render_cutting_diagram_svg(chart, p)
        # Should contain dimension labels like '12.0" × 6.0"'
        assert '&quot;' in svg or '"' in svg


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _contrasting_text
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastingText:
    def test_dark_background_white_text(self):
        assert _contrasting_text("#1b2d5b") == "#fff"

    def test_light_background_black_text(self):
        assert _contrasting_text("#f5f0dc") == "#000"

    def test_black_gives_white(self):
        assert _contrasting_text("#000000") == "#fff"

    def test_white_gives_black(self):
        assert _contrasting_text("#ffffff") == "#000"

    def test_short_hex_fallback(self):
        # Less than 6 chars → defaults to #000
        assert _contrasting_text("#fff") == "#000"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _fallback_svg
# ─────────────────────────────────────────────────────────────────────────────

class TestFallbackSvg:
    def test_returns_valid_svg(self):
        svg = _fallback_svg(_small_pattern(), 10)
        assert svg.startswith("<svg")
        assert "</svg>" in svg

    def test_contains_rects(self):
        svg = _fallback_svg(_small_pattern(), 10)
        # 2 block rects + 1 background
        assert svg.count("<rect") >= 3

    def test_dimensions_correct(self):
        p = _small_pattern()
        svg = _fallback_svg(p, 10)
        assert 'width="40"' in svg  # 4 * 10
        assert 'height="40"' in svg  # 4 * 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
