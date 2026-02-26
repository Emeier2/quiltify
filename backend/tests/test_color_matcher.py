"""Unit tests for color_matcher.py — hex-to-Kona matching via CIELAB distance."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest

from backend.services.color_matcher import (
    _hex_to_rgb,
    _rgb_to_lab,
    _color_distance,
    KonaColorMatcher,
    match_kona,
    get_palette,
    _builtin_palette,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: hex parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestHexToRgb:
    def test_full_hex(self):
        assert _hex_to_rgb("#ff8800") == (255, 136, 0)

    def test_short_hex(self):
        assert _hex_to_rgb("#f80") == (255, 136, 0)

    def test_no_hash(self):
        assert _hex_to_rgb("1b2d5b") == (27, 45, 91)

    def test_black(self):
        assert _hex_to_rgb("#000000") == (0, 0, 0)

    def test_white(self):
        assert _hex_to_rgb("#ffffff") == (255, 255, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: RGB → LAB conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestRgbToLab:
    def test_black_lab(self):
        L, a, b = _rgb_to_lab(0, 0, 0)
        assert L == pytest.approx(0.0, abs=1.0)

    def test_white_lab(self):
        L, a, b = _rgb_to_lab(255, 255, 255)
        assert L == pytest.approx(100.0, abs=1.0)

    def test_returns_three_floats(self):
        result = _rgb_to_lab(128, 64, 200)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: color distance
# ─────────────────────────────────────────────────────────────────────────────

class TestColorDistance:
    def test_same_color_zero(self):
        assert _color_distance("#ff0000", "#ff0000") == pytest.approx(0.0)

    def test_black_white_large(self):
        d = _color_distance("#000000", "#ffffff")
        assert d > 50  # should be ~100 in CIELAB L*

    def test_symmetric(self):
        d1 = _color_distance("#1b2d5b", "#c43428")
        d2 = _color_distance("#c43428", "#1b2d5b")
        assert d1 == pytest.approx(d2, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Kona matching
# ─────────────────────────────────────────────────────────────────────────────

class TestKonaMatching:
    def test_exact_match(self):
        # Navy from the builtin palette
        result = match_kona("#1b2d5b")
        assert "name" in result
        assert "hex" in result

    def test_close_match_is_navy(self):
        # Very close to Navy (#1b2d5b)
        result = match_kona("#1c2e5c")
        assert "Navy" in result["name"]

    def test_white_matches_white_or_cream(self):
        result = match_kona("#f5f5f5")
        assert any(w in result["name"] for w in ["White", "Ivory", "Cream"])

    def test_palette_loads(self):
        palette = get_palette()
        assert len(palette) >= 10
        assert all("name" in c and "hex" in c for c in palette)

    def test_builtin_palette_has_entries(self):
        palette = _builtin_palette()
        assert len(palette) == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
