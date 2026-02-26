"""
Cutting Calculator — fabric yardage math and cut-piece grouping.

A fat quarter is 18" × 22" = 396 sq in (usable ~17" × 21" after pre-wash trim).
This module computes how many fat quarters (or straight-grain cuts) are needed
per fabric given the CuttingChart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .grid_engine import CuttingChart, CutPiece, Fabric

FAT_QUARTER_WIDTH = 22.0   # inches (unwashed)
FAT_QUARTER_HEIGHT = 18.0  # inches (unwashed)
FAT_QUARTER_SQIN = FAT_QUARTER_WIDTH * FAT_QUARTER_HEIGHT  # 396
STANDARD_FABRIC_WIDTH = 44.0  # WOF (width of fabric)
WASTE_FACTOR = 1.10  # 10% for cutting waste and miscuts


@dataclass
class FabricRequirement:
    fabric_id: str
    fabric_name: str
    color_hex: str
    total_sqin: float
    fat_quarters_needed: int
    yardage_wof: float       # yards of 44" WOF fabric
    pieces: list[CutPiece]


def calculate_requirements(
    chart: CuttingChart,
    fabrics: list[Fabric],
) -> list[FabricRequirement]:
    """
    For each fabric in the cutting chart, compute how many fat quarters
    and how many yards of WOF fabric are required.
    """
    fabric_map = {f.id: f for f in fabrics}
    by_fabric = chart.by_fabric()
    requirements: list[FabricRequirement] = []

    for fabric_id, pieces in by_fabric.items():
        fab = fabric_map.get(fabric_id)
        total_sqin = sum(p.cut_width_in * p.cut_height_in * p.quantity for p in pieces)
        total_sqin_with_waste = total_sqin * WASTE_FACTOR

        fat_quarters = math.ceil(total_sqin_with_waste / FAT_QUARTER_SQIN)
        yardage = _compute_wof_yardage(pieces, STANDARD_FABRIC_WIDTH)

        requirements.append(FabricRequirement(
            fabric_id=fabric_id,
            fabric_name=fab.name if fab else fabric_id,
            color_hex=fab.color_hex if fab else "#888888",
            total_sqin=round(total_sqin, 2),
            fat_quarters_needed=max(1, fat_quarters),
            yardage_wof=yardage,
            pieces=pieces,
        ))

    return sorted(requirements, key=lambda r: r.fabric_name)


def _compute_wof_yardage(pieces: list[CutPiece], fabric_width: float) -> float:
    """
    Estimate yards of WOF fabric needed for a set of cut pieces.
    Assumes strips are cut across the width of fabric.
    """
    total_inches = 0.0
    for piece in pieces:
        # How many pieces fit across WOF?
        fits_across = math.floor(fabric_width / piece.cut_width_in)
        if fits_across == 0:
            fits_across = 1
        strips = math.ceil(piece.quantity / fits_across)
        total_inches += strips * piece.cut_height_in

    total_inches *= WASTE_FACTOR
    yards = total_inches / 36.0
    # Round up to nearest 1/8 yard
    return math.ceil(yards * 8) / 8


def format_cutting_sequence(chart: CuttingChart, fabrics: list[Fabric]) -> list[str]:
    """
    Return an ordered list of cutting instructions, sorted by fabric then by
    piece size (largest first — cut big pieces before small to use fabric efficiently).
    """
    requirements = calculate_requirements(chart, fabrics)
    instructions: list[str] = []

    for req in requirements:
        sorted_pieces = sorted(
            req.pieces,
            key=lambda p: p.cut_width_in * p.cut_height_in,
            reverse=True,
        )
        instructions.append(f"### {req.fabric_name} ({req.color_hex})")
        instructions.append(
            f"Total needed: ~{req.total_sqin:.0f} sq in "
            f"({req.fat_quarters_needed} fat quarter{'s' if req.fat_quarters_needed > 1 else ''} "
            f"or {req.yardage_wof} yd WOF)"
        )
        for piece in sorted_pieces:
            instructions.append(
                f"  • Cut {piece.quantity}× pieces {piece.cut_width_in}\" × {piece.cut_height_in}\""
            )

    return instructions
