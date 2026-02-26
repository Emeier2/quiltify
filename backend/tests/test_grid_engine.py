"""Unit tests for grid_engine.py — the quilting domain model."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import math
import pytest

from backend.services.grid_engine import Fabric, Block, QuiltPattern, CuttingChart, CutPiece


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def make_6_fabric_40x50_pattern() -> QuiltPattern:
    """
    40×50 grid, 2.5" finished block, 0.25" seam → 3.0" cut.
    6 fabrics, horizontal stripes of equal height (~8-9 rows each).
    """
    fabrics = [
        Fabric(id=f"f{i}", color_hex=f"#{'ab' * 3}", name=f"Fabric {i}")
        for i in range(1, 7)
    ]
    colors = ["#1b2d5b", "#c43428", "#f5f0dc", "#4a7c3f", "#d4a42a", "#7db8d8"]
    for i, f in enumerate(fabrics):
        f.color_hex = colors[i]

    blocks = []
    stripe_h = 50 // 6  # 8 rows each; last fabric gets remainder
    for i, fab in enumerate(fabrics):
        y_start = i * stripe_h
        y_end = y_start + stripe_h if i < 5 else 50
        blocks.append(Block(x=0, y=y_start, width=40, height=y_end - y_start,
                            fabric_id=fab.id))

    return QuiltPattern(
        grid_width=40,
        grid_height=50,
        block_size_in=2.5,
        seam_allowance=0.25,
        fabrics=fabrics,
        blocks=blocks,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Fabric
# ─────────────────────────────────────────────────────────────────────────────

class TestFabric:
    def test_fat_quarters_small(self):
        f = Fabric(id="f1", color_hex="#fff", name="Test", total_sqin=300.0)
        # 300 * 1.1 = 330, 396 per FQ → ceil(330/396) = 1
        assert f.fat_quarters() == 1

    def test_fat_quarters_larger(self):
        f = Fabric(id="f1", color_hex="#fff", name="Test", total_sqin=800.0)
        # 800 * 1.1 = 880, 880/396 = 2.22 → ceil = 3
        assert f.fat_quarters() == 3

    def test_fat_quarters_exact(self):
        f = Fabric(id="f1", color_hex="#fff", name="Test", total_sqin=396.0 / 1.1)
        # 360 * 1.1 = 396 → ceil(396/396) = 1
        assert f.fat_quarters() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Block
# ─────────────────────────────────────────────────────────────────────────────

class TestBlock:
    def test_area(self):
        b = Block(x=0, y=0, width=3, height=4, fabric_id="f1")
        assert b.area_cells() == 12

    def test_cells(self):
        b = Block(x=2, y=1, width=2, height=2, fabric_id="f1")
        cells = set(b.cells())
        assert cells == {(2, 1), (3, 1), (2, 2), (3, 2)}

    def test_single_cell(self):
        b = Block(x=5, y=7, width=1, height=1, fabric_id="f1")
        assert b.cells() == [(5, 7)]
        assert b.area_cells() == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: QuiltPattern validation
# ─────────────────────────────────────────────────────────────────────────────

class TestQuiltPatternValidation:
    def test_valid_pattern_no_errors(self):
        p = make_6_fabric_40x50_pattern()
        errors = p.validate()
        assert errors == [], f"Unexpected errors: {errors}"

    def test_out_of_bounds_x(self):
        p = make_6_fabric_40x50_pattern()
        p.blocks[0].width = 45  # exceeds 40
        errors = p.validate()
        assert any("X bounds" in e for e in errors)

    def test_out_of_bounds_y(self):
        p = make_6_fabric_40x50_pattern()
        p.blocks[-1].height = 20  # would exceed 50
        errors = p.validate()
        assert any("Y bounds" in e or "uncovered" in e for e in errors)

    def test_overlap_detected(self):
        p = make_6_fabric_40x50_pattern()
        # Add a block overlapping the first stripe
        p.blocks.append(Block(x=0, y=0, width=5, height=5, fabric_id="f1"))
        errors = p.validate()
        assert any("Overlap" in e for e in errors)

    def test_unknown_fabric(self):
        p = make_6_fabric_40x50_pattern()
        p.blocks[0].fabric_id = "unknown_id"
        errors = p.validate()
        assert any("unknown fabric_id" in e for e in errors)

    def test_uncovered_cells(self):
        p = QuiltPattern(grid_width=4, grid_height=4, block_size_in=2.5,
                         fabrics=[Fabric(id="f1", color_hex="#fff", name="T")])
        # Only add a partial block
        p.blocks.append(Block(x=0, y=0, width=4, height=3, fabric_id="f1"))
        errors = p.validate()
        assert any("uncovered" in e for e in errors)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Area calculation + cutting chart
# ─────────────────────────────────────────────────────────────────────────────

class TestCuttingChart:
    def test_cut_size(self):
        p = make_6_fabric_40x50_pattern()
        # 2.5" finished + 2 × 0.25" seam = 3.0"
        assert p.cut_size_in == pytest.approx(3.0, abs=1e-6)

    def test_total_cells(self):
        p = make_6_fabric_40x50_pattern()
        total = sum(b.area_cells() for b in p.blocks)
        assert total == 40 * 50

    def test_total_square_inches(self):
        p = make_6_fabric_40x50_pattern()
        p.compute_fabric_areas()
        total_sqin = sum(f.total_sqin for f in p.fabrics)
        # Each cell = 3.0" × 3.0" = 9 sq in; 40×50 = 2000 cells
        expected = 2000 * 9.0
        assert total_sqin == pytest.approx(expected, rel=1e-4)

    def test_fat_quarter_counts(self):
        p = make_6_fabric_40x50_pattern()
        p.compute_fabric_areas()
        for fab in p.fabrics:
            fq = fab.fat_quarters()
            # Each fabric covers either 8 or 10 rows × 40 cols = 320–400 cells
            # 320 cells × 9 = 2880 sq in × 1.1 / 396 ≈ 7.99 → 8 FQ
            # 400 cells × 9 = 3600 sq in × 1.1 / 396 ≈ 10.0 → 10 FQ
            assert 1 <= fq <= 20, f"Unexpected fat quarters {fq} for {fab.name}"

    def test_cutting_chart_has_entries(self):
        p = make_6_fabric_40x50_pattern()
        chart = p.to_cutting_chart()
        assert len(chart.pieces) == 6  # one piece type per fabric (each is a full-width stripe)

    def test_cutting_chart_quantities(self):
        p = make_6_fabric_40x50_pattern()
        chart = p.to_cutting_chart()
        total_pieces = sum(piece.quantity for piece in chart.pieces)
        assert total_pieces == 6  # one block per fabric

    def test_cutting_chart_dimensions(self):
        p = make_6_fabric_40x50_pattern()
        chart = p.to_cutting_chart()
        for piece in chart.pieces:
            # Width should be 40 × 3.0" = 120.0"
            assert piece.cut_width_in == pytest.approx(120.0, abs=0.01) or \
                   piece.cut_height_in == pytest.approx(120.0, abs=0.01)

    def test_cut_size_in_chart(self):
        p = QuiltPattern(
            grid_width=10, grid_height=10,
            block_size_in=2.5, seam_allowance=0.25,
            fabrics=[Fabric(id="f1", color_hex="#fff", name="White")],
            blocks=[Block(x=0, y=0, width=10, height=10, fabric_id="f1")],
        )
        chart = p.to_cutting_chart()
        assert chart.cut_size_in == pytest.approx(3.0, abs=1e-6)

    def test_by_fabric_groups(self):
        p = make_6_fabric_40x50_pattern()
        chart = p.to_cutting_chart()
        by_fab = chart.by_fabric()
        assert len(by_fab) == 6
        for fab_id, pieces in by_fab.items():
            assert all(pc.fabric_id == fab_id for pc in pieces)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Serialization roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        p = make_6_fabric_40x50_pattern()
        d = p.to_dict()
        p2 = QuiltPattern.from_dict(d)
        assert p2.grid_width == p.grid_width
        assert p2.grid_height == p.grid_height
        assert len(p2.fabrics) == len(p.fabrics)
        assert len(p2.blocks) == len(p.blocks)

    def test_dict_has_finished_size(self):
        p = make_6_fabric_40x50_pattern()
        d = p.to_dict()
        # 40 × 2.5 = 100", 50 × 2.5 = 125"
        assert d["finished_width_in"] == pytest.approx(100.0)
        assert d["finished_height_in"] == pytest.approx(125.0)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Grid extractor (synthetic fallback, no image libs required)
# ─────────────────────────────────────────────────────────────────────────────

class TestGridExtractorFallback:
    def test_synthetic_fallback_valid(self):
        from backend.services.grid_extractor import _synthetic_fallback
        p = _synthetic_fallback(40, 50, 6, 2.5, 0.25)
        errors = p.validate()
        assert errors == [], f"Synthetic fallback has errors: {errors}"

    def test_synthetic_fallback_fabric_count(self):
        from backend.services.grid_extractor import _synthetic_fallback
        p = _synthetic_fallback(40, 50, 4, 2.5, 0.25)
        assert len(p.fabrics) == 4

    def test_synthetic_fallback_covers_all_cells(self):
        from backend.services.grid_extractor import _synthetic_fallback
        p = _synthetic_fallback(20, 30, 6, 2.5, 0.25)
        covered = p.covered_cells()
        assert covered == p.all_cells()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
