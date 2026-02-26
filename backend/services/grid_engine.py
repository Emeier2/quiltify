"""
Grid Engine — core quilting domain model.

QuiltPattern holds a grid of cells (each cell belongs to a Block),
validates geometry, and produces a CuttingChart with precise dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Fabric:
    id: str
    color_hex: str
    name: str
    total_sqin: float = 0.0

    def fat_quarters(self, seam_allowance: float = 0.25) -> float:
        """Returns number of fat quarters needed (18" x 22" = 396 sq in)."""
        # Add 10% for cutting waste
        return math.ceil((self.total_sqin * 1.1) / 396)

    def yardage(self, cut_width: float, cut_height: float, quantity: int,
                fabric_width: float = 44.0) -> float:
        """Compute yards needed for a given cut piece run."""
        cuts_per_strip = math.floor(fabric_width / cut_width)
        if cuts_per_strip == 0:
            cuts_per_strip = 1
        strips_needed = math.ceil(quantity / cuts_per_strip)
        total_length_in = strips_needed * cut_height
        return round(total_length_in / 36, 2)


@dataclass
class Block:
    x: int           # grid column (0-indexed)
    y: int           # grid row (0-indexed)
    width: int       # in grid units
    height: int      # in grid units
    fabric_id: str

    def cells(self) -> list[tuple[int, int]]:
        return [(self.x + dx, self.y + dy)
                for dy in range(self.height)
                for dx in range(self.width)]

    def area_cells(self) -> int:
        return self.width * self.height


@dataclass
class CutPiece:
    fabric_id: str
    fabric_name: str
    color_hex: str
    cut_width_in: float
    cut_height_in: float
    quantity: int

    def label(self) -> str:
        return f'{self.cut_width_in}" × {self.cut_height_in}" — qty {self.quantity}'


@dataclass
class CuttingChart:
    block_size_in: float
    seam_allowance: float
    pieces: list[CutPiece] = field(default_factory=list)

    @property
    def cut_size_in(self) -> float:
        return round(self.block_size_in + 2 * self.seam_allowance, 4)

    def total_pieces(self) -> int:
        return sum(p.quantity for p in self.pieces)

    def by_fabric(self) -> dict[str, list[CutPiece]]:
        result: dict[str, list[CutPiece]] = {}
        for piece in self.pieces:
            result.setdefault(piece.fabric_id, []).append(piece)
        return result


@dataclass
class QuiltPattern:
    grid_width: int = 40
    grid_height: int = 50
    block_size_in: float = 2.5
    seam_allowance: float = 0.25
    fabrics: list[Fabric] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Computed properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def cut_size_in(self) -> float:
        return round(self.block_size_in + 2 * self.seam_allowance, 4)

    @property
    def finished_width_in(self) -> float:
        return self.grid_width * self.block_size_in

    @property
    def finished_height_in(self) -> float:
        return self.grid_height * self.block_size_in

    @property
    def fabric_map(self) -> dict[str, Fabric]:
        return {f.id: f for f in self.fabrics}

    # ------------------------------------------------------------------ #
    # Grid helpers                                                         #
    # ------------------------------------------------------------------ #

    def cell_grid(self) -> dict[tuple[int, int], str]:
        """Return a dict mapping (x, y) → fabric_id for every assigned cell."""
        grid: dict[tuple[int, int], str] = {}
        for block in self.blocks:
            for cell in block.cells():
                grid[cell] = block.fabric_id
        return grid

    def covered_cells(self) -> set[tuple[int, int]]:
        return set(self.cell_grid().keys())

    def all_cells(self) -> set[tuple[int, int]]:
        return {(x, y)
                for x in range(self.grid_width)
                for y in range(self.grid_height)}

    # ------------------------------------------------------------------ #
    # Validation                                                           #
    # ------------------------------------------------------------------ #

    def validate(self) -> list[str]:
        errors: list[str] = []
        fabric_ids = {f.id for f in self.fabrics}

        seen: dict[tuple[int, int], int] = {}
        for i, block in enumerate(self.blocks):
            if block.x < 0 or block.x + block.width > self.grid_width:
                errors.append(
                    f"Block {i} (fabric={block.fabric_id}) out of X bounds: "
                    f"x={block.x}, width={block.width}, grid_width={self.grid_width}")
            if block.y < 0 or block.y + block.height > self.grid_height:
                errors.append(
                    f"Block {i} (fabric={block.fabric_id}) out of Y bounds: "
                    f"y={block.y}, height={block.height}, grid_height={self.grid_height}")
            if block.fabric_id not in fabric_ids:
                errors.append(
                    f"Block {i} references unknown fabric_id='{block.fabric_id}'")
            for cell in block.cells():
                if cell in seen:
                    errors.append(
                        f"Overlap at cell {cell} between block {seen[cell]} and block {i}")
                else:
                    seen[cell] = i

        uncovered = self.all_cells() - set(seen.keys())
        if uncovered:
            errors.append(f"{len(uncovered)} cells uncovered (e.g. {sorted(uncovered)[:5]})")

        return errors

    # ------------------------------------------------------------------ #
    # Fabric area calculation                                              #
    # ------------------------------------------------------------------ #

    def compute_fabric_areas(self) -> None:
        """Update each Fabric.total_sqin based on block assignments."""
        sq_per_cell = self.cut_size_in ** 2
        area: dict[str, float] = {f.id: 0.0 for f in self.fabrics}
        for block in self.blocks:
            area[block.fabric_id] = area.get(block.fabric_id, 0.0) + (
                block.area_cells() * sq_per_cell
            )
        for fabric in self.fabrics:
            fabric.total_sqin = round(area[fabric.id], 2)

    # ------------------------------------------------------------------ #
    # Cutting chart                                                        #
    # ------------------------------------------------------------------ #

    def to_cutting_chart(self) -> CuttingChart:
        """
        Group blocks by fabric_id + (width, height) → CutPiece entries.
        Each CutPiece represents one type of rectangular cut for one fabric.
        """
        self.compute_fabric_areas()
        fabric_map = self.fabric_map

        # key: (fabric_id, cut_width_in, cut_height_in) → count
        piece_counts: dict[tuple[str, float, float], int] = {}

        for block in self.blocks:
            w_in = round(block.width * self.cut_size_in, 4)
            h_in = round(block.height * self.cut_size_in, 4)
            # Normalize so width <= height for consistent grouping
            if w_in > h_in:
                w_in, h_in = h_in, w_in
            key = (block.fabric_id, w_in, h_in)
            piece_counts[key] = piece_counts.get(key, 0) + 1

        chart = CuttingChart(
            block_size_in=self.block_size_in,
            seam_allowance=self.seam_allowance,
        )
        for (fabric_id, w_in, h_in), qty in sorted(piece_counts.items()):
            fab = fabric_map.get(fabric_id)
            chart.pieces.append(CutPiece(
                fabric_id=fabric_id,
                fabric_name=fab.name if fab else fabric_id,
                color_hex=fab.color_hex if fab else "#888888",
                cut_width_in=w_in,
                cut_height_in=h_in,
                quantity=qty,
            ))
        return chart

    # ------------------------------------------------------------------ #
    # Serialization                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        self.compute_fabric_areas()
        return {
            "grid_width": self.grid_width,
            "grid_height": self.grid_height,
            "block_size_in": self.block_size_in,
            "seam_allowance": self.seam_allowance,
            "finished_width_in": self.finished_width_in,
            "finished_height_in": self.finished_height_in,
            "fabrics": [
                {"id": f.id, "color_hex": f.color_hex, "name": f.name,
                 "total_sqin": f.total_sqin, "fat_quarters": f.fat_quarters(self.seam_allowance)}
                for f in self.fabrics
            ],
            "blocks": [
                {"x": b.x, "y": b.y, "width": b.width, "height": b.height,
                 "fabric_id": b.fabric_id}
                for b in self.blocks
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuiltPattern":
        pattern = cls(
            grid_width=data["grid_width"],
            grid_height=data["grid_height"],
            block_size_in=data.get("block_size_in", 2.5),
            seam_allowance=data.get("seam_allowance", 0.25),
        )
        for f in data.get("fabrics", []):
            pattern.fabrics.append(Fabric(
                id=f["id"],
                color_hex=f["color_hex"],
                name=f["name"],
                total_sqin=f.get("total_sqin", 0.0),
            ))
        for b in data.get("blocks", []):
            pattern.blocks.append(Block(
                x=b["x"], y=b["y"],
                width=b["width"], height=b["height"],
                fabric_id=b["fabric_id"],
            ))
        return pattern
