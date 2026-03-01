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
    40×50 grid, 2.5" finished cell, 0.25" seam → 3.0" cut.
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

    cell_sizes = uniform_cell_sizes(40, 50, 2.5)
    return QuiltPattern(
        grid_width=40,
        grid_height=50,
        quilt_width_in=100.0,
        quilt_height_in=125.0,
        seam_allowance=0.25,
        fabrics=fabrics,
        blocks=blocks,
        cell_sizes=cell_sizes,
    )


def uniform_cell_sizes(grid_width: int, grid_height: int, size: float) -> list[dict[str, float]]:
    return [{"w": size, "h": size} for _ in range(grid_width * grid_height)]


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
        p = QuiltPattern(grid_width=4, grid_height=4, quilt_width_in=10.0, quilt_height_in=10.0,
                         fabrics=[Fabric(id="f1", color_hex="#fff", name="T")],
                         cell_sizes=uniform_cell_sizes(4, 4, 2.5))
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
        w, h = p.cell_size_at(0, 0)
        assert (w + 2 * p.seam_allowance) == pytest.approx(3.0, abs=1e-6)

    def test_total_cells(self):
        p = make_6_fabric_40x50_pattern()
        total = sum(b.area_cells() for b in p.blocks)
        assert total == 40 * 50

    def test_total_square_inches(self):
        p = make_6_fabric_40x50_pattern()
        p.compute_fabric_areas()
        total_sqin = sum(f.total_sqin for f in p.fabrics)
        # Each cell = 2.5" × 2.5" = 6.25 sq in; 40×50 = 2000 cells
        expected = 2000 * 6.25
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
            # Width should be 40 × 2.5" + 0.5" = 100.5"
            assert piece.cut_width_in == pytest.approx(100.5, abs=0.01) or \
                   piece.cut_height_in == pytest.approx(100.5, abs=0.01)

    def test_cut_size_in_chart(self):
        p = QuiltPattern(
            grid_width=10, grid_height=10,
            quilt_width_in=25.0, quilt_height_in=25.0, seam_allowance=0.25,
            fabrics=[Fabric(id="f1", color_hex="#fff", name="White")],
            blocks=[Block(x=0, y=0, width=10, height=10, fabric_id="f1")],
            cell_sizes=uniform_cell_sizes(10, 10, 2.5),
        )
        chart = p.to_cutting_chart()
        piece = chart.pieces[0]
        assert piece.cut_width_in == pytest.approx(25.5, abs=1e-6) or \
               piece.cut_height_in == pytest.approx(25.5, abs=1e-6)

    def test_by_fabric_groups(self):
        p = make_6_fabric_40x50_pattern()
        chart = p.to_cutting_chart()
        by_fab = chart.by_fabric()
        assert len(by_fab) == 6
        for fab_id, pieces in by_fab.items():
            assert all(pc.fabric_id == fab_id for pc in pieces)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Corner-aware cutting chart
# ─────────────────────────────────────────────────────────────────────────────

class TestCornerCuttingChart:
    def _pattern_with_corners(self) -> QuiltPattern:
        """4x4 grid, 2.5" cells, 2 fabrics. One block has 2 corners."""
        fabrics = [
            Fabric(id="f1", color_hex="#1b2d5b", name="Navy"),
            Fabric(id="f2", color_hex="#c43428", name="Red"),
        ]
        blocks = [
            Block(x=0, y=0, width=4, height=2, fabric_id="f1",
                  corners={"ne": "f2", "se": "f2"}),
            Block(x=0, y=2, width=4, height=2, fabric_id="f2"),
        ]
        return QuiltPattern(
            grid_width=4, grid_height=4,
            quilt_width_in=10.0, quilt_height_in=10.0, seam_allowance=0.25,
            fabrics=fabrics, blocks=blocks,
            cell_sizes=uniform_cell_sizes(4, 4, 2.5),
        )

    def test_corner_squares_in_chart(self):
        """Cutting chart should include corner square pieces."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        corner_pieces = [pc for pc in chart.pieces if pc.piece_type == "corner"]
        assert len(corner_pieces) >= 1

    def test_corner_square_fabric_is_corner_fabric(self):
        """Corner squares should belong to the corner fabric, not the base."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        corner_pieces = [pc for pc in chart.pieces if pc.piece_type == "corner"]
        for cp in corner_pieces:
            assert cp.fabric_id == "f2"

    def test_corner_square_is_square(self):
        """Corner square cut pieces should be square (w == h)."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        corner_pieces = [pc for pc in chart.pieces if pc.piece_type == "corner"]
        for cp in corner_pieces:
            assert cp.cut_width_in == cp.cut_height_in

    def test_corner_square_size(self):
        """Corner square = max(cell_w, cell_h) + 2 * seam_allowance."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        corner_pieces = [pc for pc in chart.pieces if pc.piece_type == "corner"]
        # Cell is 2.5" x 2.5", so corner square = 2.5 + 0.5 = 3.0"
        for cp in corner_pieces:
            assert cp.cut_width_in == pytest.approx(3.0, abs=1e-4)

    def test_corner_square_quantity(self):
        """Two corners on one block → 2 corner squares total."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        corner_pieces = [pc for pc in chart.pieces if pc.piece_type == "corner"]
        total_corners = sum(cp.quantity for cp in corner_pieces)
        assert total_corners == 2

    def test_base_and_corner_pieces_both_present(self):
        """Chart should have both base rectangles and corner squares."""
        p = self._pattern_with_corners()
        chart = p.to_cutting_chart()
        types = {pc.piece_type for pc in chart.pieces}
        assert "base" in types
        assert "corner" in types

    def test_corner_area_split(self):
        """Fabric areas should reflect 50/50 split for corner cells."""
        p = self._pattern_with_corners()
        p.compute_fabric_areas()
        navy = next(f for f in p.fabrics if f.id == "f1")
        red = next(f for f in p.fabrics if f.id == "f2")
        # Navy has 4x2=8 cells, but 2 corner cells lose 50% each
        # Navy area: 6 full cells + 2 half cells = 7 cells worth = 7 * 6.25 = 43.75
        # Red area: 8 full cells + 2 half cells = 9 cells worth = 9 * 6.25 = 56.25
        assert navy.total_sqin == pytest.approx(43.75, abs=0.1)
        assert red.total_sqin == pytest.approx(56.25, abs=0.1)


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
        p = _synthetic_fallback(40, 50, 6, 100.0, 125.0, 0.25)
        errors = p.validate()
        assert errors == [], f"Synthetic fallback has errors: {errors}"

    def test_synthetic_fallback_fabric_count(self):
        from backend.services.grid_extractor import _synthetic_fallback
        p = _synthetic_fallback(40, 50, 4, 100.0, 125.0, 0.25)
        assert len(p.fabrics) == 4

    def test_synthetic_fallback_covers_all_cells(self):
        from backend.services.grid_extractor import _synthetic_fallback
        p = _synthetic_fallback(20, 30, 6, 50.0, 75.0, 0.25)
        covered = p.covered_cells()
        assert covered == p.all_cells()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
