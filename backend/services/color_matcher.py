"""
Color Matcher — maps arbitrary hex colors to named Kona Cotton palette entries.

Uses Euclidean distance in CIELAB color space for perceptually accurate matching.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

DATA_PATH = Path(__file__).parent.parent / "data" / "kona_cotton_palette.json"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to CIELAB."""
    # Normalize to [0, 1]
    r_, g_, b_ = r / 255.0, g / 255.0, b / 255.0

    # Linearize
    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rl, gl, bl = linearize(r_), linearize(g_), linearize(b_)

    # RGB → XYZ (D65)
    x = rl * 0.4124564 + gl * 0.3575761 + bl * 0.1804375
    y = rl * 0.2126729 + gl * 0.7151522 + bl * 0.0721750
    z = rl * 0.0193339 + gl * 0.1191920 + bl * 0.9503041

    # Normalize by D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    def f(t: float) -> float:
        return t ** (1 / 3) if t > 0.008856 else (7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_val = 200 * (fy - fz)
    return L, a, b_val


def _color_distance(hex1: str, hex2: str) -> float:
    """CIELAB Euclidean distance between two hex colors."""
    L1, a1, b1 = _rgb_to_lab(*_hex_to_rgb(hex1))
    L2, a2, b2 = _rgb_to_lab(*_hex_to_rgb(hex2))
    return math.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)


class KonaColorMatcher:
    def __init__(self) -> None:
        self._palette: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if DATA_PATH.exists():
            self._palette = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        else:
            self._palette = _builtin_palette()
        self._loaded = True

    def match(self, hex_color: str) -> dict:
        """Return the closest Kona Cotton color entry {name, hex}."""
        self._ensure_loaded()
        best = min(self._palette, key=lambda c: _color_distance(hex_color, c["hex"]))
        return best

    def match_name(self, hex_color: str) -> str:
        return self.match(hex_color)["name"]

    def palette(self) -> list[dict]:
        self._ensure_loaded()
        return self._palette


_matcher = KonaColorMatcher()


def match_kona(hex_color: str) -> dict:
    return _matcher.match(hex_color)


def get_palette() -> list[dict]:
    return _matcher.palette()


def _builtin_palette() -> list[dict]:
    """Small fallback palette when the JSON file is missing."""
    return [
        {"name": "Kona Cotton - Black", "hex": "#1a1a1a"},
        {"name": "Kona Cotton - White", "hex": "#f5f5f5"},
        {"name": "Kona Cotton - Cream", "hex": "#f5f0dc"},
        {"name": "Kona Cotton - Navy", "hex": "#1b2d5b"},
        {"name": "Kona Cotton - Cobalt", "hex": "#2c5fa6"},
        {"name": "Kona Cotton - Sky", "hex": "#7db8d8"},
        {"name": "Kona Cotton - Grass", "hex": "#4a7c3f"},
        {"name": "Kona Cotton - Lime", "hex": "#98c44a"},
        {"name": "Kona Cotton - Tomato", "hex": "#c43428"},
        {"name": "Kona Cotton - Tangerine", "hex": "#e87535"},
        {"name": "Kona Cotton - Gold", "hex": "#d4a42a"},
        {"name": "Kona Cotton - Sand", "hex": "#c8b585"},
        {"name": "Kona Cotton - Chocolate", "hex": "#5e3a1e"},
        {"name": "Kona Cotton - Charcoal", "hex": "#4a4a4a"},
        {"name": "Kona Cotton - Fog", "hex": "#9eadb5"},
        {"name": "Kona Cotton - Dusty Blue", "hex": "#6b8fa8"},
        {"name": "Kona Cotton - Sage", "hex": "#8a9e7e"},
        {"name": "Kona Cotton - Khaki", "hex": "#c4b98a"},
        {"name": "Kona Cotton - Mushroom", "hex": "#a08070"},
        {"name": "Kona Cotton - Eggplant", "hex": "#4a2060"},
        {"name": "Kona Cotton - Bordeaux", "hex": "#7a1f2e"},
        {"name": "Kona Cotton - Teal", "hex": "#2a7a6e"},
        {"name": "Kona Cotton - Aqua", "hex": "#40b4b0"},
        {"name": "Kona Cotton - Maize", "hex": "#f0c040"},
        {"name": "Kona Cotton - Coral", "hex": "#e8705a"},
        {"name": "Kona Cotton - Rose", "hex": "#d87090"},
        {"name": "Kona Cotton - Lavender", "hex": "#a080c0"},
        {"name": "Kona Cotton - Periwinkle", "hex": "#7080c0"},
        {"name": "Kona Cotton - Ivory", "hex": "#f8f0e0"},
        {"name": "Kona Cotton - Natural", "hex": "#e8dcc0"},
    ]
