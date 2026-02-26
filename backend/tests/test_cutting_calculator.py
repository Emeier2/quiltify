"""Unit tests for cutting_calculator.py — yardage math and cutting sequences."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest

from backend.services.grid_engine import Fabric, Block, QuiltPattern, CutPiece, CuttingChart
from backend.services.cutting_calculator import (
    calculate_requirements,
    format_cutting_sequence,
    _compute_wof_yardage,
    FabricRequirement,
    FAT_QUARTER_SQIN,
    WASTE_FACTOR,
)


def _simple_pattern() -> QuiltPattern:
    """10x10 grid, 2 fabrics, each covering half."""
    fabrics = [
        Fabric(id="f1", color_hex="#1b2d5b", name="Navy"),
        Fabric(id="f2", color_hex="#c43428", name="Red"),
    ]
    blocks = [
        Block(x=0, y=0, width=10, height=5, fabric_id="f1"),
        Block(x=0, y=5, width=10, height=5, fabric_id="f2"),
    ]
    return QuiltPattern(
        grid_width=10, grid_height=10,
        block_size_in=2.5, seam_allowance=0.25,
        fabrics=fabrics, blocks=blocks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: calculate_requirements
# ─────────────────────────────────────────────────────────────────────────────

class TestCalculateRequirements:
    def test_returns_one_per_fabric(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        reqs = calculate_requirements(chart, p.fabrics)
        assert len(reqs) == 2

    def test_total_sqin_positive(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        reqs = calculate_requirements(chart, p.fabrics)
        for req in reqs:
            assert req.total_sqin > 0

    def test_fat_quarters_at_least_one(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        reqs = calculate_requirements(chart, p.fabrics)
        for req in reqs:
            assert req.fat_quarters_needed >= 1

    def test_yardage_positive(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        reqs = calculate_requirements(chart, p.fabrics)
        for req in reqs:
            assert req.yardage_wof > 0

    def test_sorted_by_fabric_name(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        reqs = calculate_requirements(chart, p.fabrics)
        names = [r.fabric_name for r in reqs]
        assert names == sorted(names)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: format_cutting_sequence
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatCuttingSequence:
    def test_returns_list_of_strings(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        instructions = format_cutting_sequence(chart, p.fabrics)
        assert isinstance(instructions, list)
        assert all(isinstance(s, str) for s in instructions)

    def test_contains_fabric_names(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        instructions = format_cutting_sequence(chart, p.fabrics)
        text = "\n".join(instructions)
        assert "Navy" in text
        assert "Red" in text

    def test_contains_cut_dimensions(self):
        p = _simple_pattern()
        chart = p.to_cutting_chart()
        instructions = format_cutting_sequence(chart, p.fabrics)
        text = "\n".join(instructions)
        # Should mention piece dimensions with "×"
        assert "×" in text or "x" in text.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _compute_wof_yardage
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeWofYardage:
    def test_single_small_piece(self):
        pieces = [CutPiece(
            fabric_id="f1", fabric_name="T", color_hex="#fff",
            cut_width_in=3.0, cut_height_in=3.0, quantity=1,
        )]
        yards = _compute_wof_yardage(pieces, 44.0)
        assert yards > 0
        # 1 piece of 3"×3" should need very little yardage
        assert yards <= 0.25

    def test_many_pieces_needs_more(self):
        pieces = [CutPiece(
            fabric_id="f1", fabric_name="T", color_hex="#fff",
            cut_width_in=3.0, cut_height_in=3.0, quantity=100,
        )]
        yards = _compute_wof_yardage(pieces, 44.0)
        assert yards > 0.5

    def test_wide_piece_exceeding_wof(self):
        pieces = [CutPiece(
            fabric_id="f1", fabric_name="T", color_hex="#fff",
            cut_width_in=50.0, cut_height_in=3.0, quantity=1,
        )]
        yards = _compute_wof_yardage(pieces, 44.0)
        # Piece wider than fabric — still should compute without error
        assert yards > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
