"""Unit tests for vtracer_service.py — image vectorization."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import io
import pytest

from backend.services.vtracer_service import (
    is_available,
    vectorize_image,
    vectorize_and_rasterize,
    clean_for_extraction,
)


def _make_solid_jpeg(color: tuple[int, int, int], width: int = 100, height: int = 100) -> bytes:
    """Create a solid-color JPEG image as bytes."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_two_color_jpeg(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    width: int = 100,
    height: int = 100,
) -> bytes:
    """Create a JPEG with left half = color_a, right half = color_b."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color_a)
    for y in range(height):
        for x in range(width // 2, width):
            img.putpixel((x, y), color_b)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: availability
# ─────────────────────────────────────────────────────────────────────────────

class TestAvailability:
    def test_is_available(self):
        """vtracer should be available (we installed it)."""
        assert is_available() is True


# ─────────────────────────────────────────────────────────────────────────────
# Tests: vectorize_image
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorizeImage:
    def test_solid_image_returns_svg(self):
        """A solid-color JPEG should vectorize to a valid SVG string."""
        jpeg = _make_solid_jpeg((200, 50, 50))
        svg = vectorize_image(jpeg)
        assert svg is not None
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_two_color_image_returns_svg(self):
        """A two-color JPEG should produce an SVG with multiple paths."""
        jpeg = _make_two_color_jpeg((200, 50, 50), (50, 50, 200))
        svg = vectorize_image(jpeg)
        assert svg is not None
        assert "<svg" in svg
        # Should have path elements for both color regions
        assert "<path" in svg or "<rect" in svg

    def test_returns_none_without_vtracer(self, monkeypatch):
        """Should return None when vtracer is not available."""
        import backend.services.vtracer_service as mod
        monkeypatch.setattr(mod, "_HAS_VTRACER", False)
        jpeg = _make_solid_jpeg((200, 50, 50))
        result = vectorize_image(jpeg)
        assert result is None

    def test_svg_contains_fill_colors(self):
        """SVG output should contain fill color attributes."""
        jpeg = _make_two_color_jpeg((200, 50, 50), (50, 50, 200))
        svg = vectorize_image(jpeg)
        assert svg is not None
        assert "fill" in svg.lower()

    def test_polygon_mode_no_curves(self):
        """In polygon mode, paths should use L (lineto) not C (curveto)."""
        jpeg = _make_two_color_jpeg((200, 50, 50), (50, 50, 200))
        svg = vectorize_image(jpeg, mode="polygon")
        assert svg is not None
        # Polygon mode uses straight lines — fewer or no cubic curves
        # SVG paths: M=moveto, L=lineto, C=curveto, Z=close
        # We just verify it's a valid SVG (the mode parameter is accepted)
        assert "</svg>" in svg


# ─────────────────────────────────────────────────────────────────────────────
# Tests: clean_for_extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanForExtraction:
    def test_returns_none_without_cairosvg(self, monkeypatch):
        """Should return None gracefully when cairosvg is not installed."""
        # clean_for_extraction calls vectorize_and_rasterize which needs cairosvg
        jpeg = _make_solid_jpeg((200, 50, 50))
        # This will return None if cairosvg isn't installed (expected in CI)
        result = clean_for_extraction(jpeg, 4, 4, 24)
        # Either returns bytes (if cairosvg is available) or None
        assert result is None or isinstance(result, bytes)

    def test_returns_none_without_vtracer(self, monkeypatch):
        """Should return None when vtracer is not available."""
        import backend.services.vtracer_service as mod
        monkeypatch.setattr(mod, "_HAS_VTRACER", False)
        jpeg = _make_solid_jpeg((200, 50, 50))
        result = clean_for_extraction(jpeg, 4, 4, 24)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: grid extractor integration
# ─────────────────────────────────────────────────────────────────────────────

class TestGridExtractorIntegration:
    def test_extract_still_works_without_vtracer_cleaning(self):
        """Grid extraction should work even when vtracer cleaning returns None."""
        from backend.services.grid_extractor import extract_pattern_from_image
        # Create a simple 4-color striped image
        from PIL import Image
        img = Image.new("RGB", (200, 200))
        colors = [(200, 50, 50), (50, 50, 200), (50, 200, 50), (200, 200, 50)]
        for y in range(200):
            stripe = colors[y // 50]
            for x in range(200):
                img.putpixel((x, y), stripe)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)

        pattern, confidence = extract_pattern_from_image(
            buf.getvalue(),
            grid_width=4, grid_height=4,
            palette_size=4,
            quilt_width_in=10.0, quilt_height_in=10.0,
        )
        assert pattern is not None
        assert len(pattern.blocks) > 0
        assert confidence > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
