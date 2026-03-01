"""
Grid Engine - core quilting domain model.

QuiltPattern holds a grid of cells (each cell belongs to a Block),
validates geometry, and produces a CuttingChart with precise dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


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
    # Optional stitch-and-flip corners: {"nw": "f2", "ne": "f3", ...}
    corners: dict[str, str] = field(default_factory=dict)

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
    # "base" for rectangle pieces, "corner" for stitch-and-flip squares
    piece_type: str = "base"

    def label(self) -> str:
        return f'{self.cut_width_in}" x {self.cut_height_in}" - qty {self.quantity}'


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
    quilt_width_in: float = 60.0
    quilt_height_in: float = 72.0
    seam_allowance: float = 0.25
    fabrics: list[Fabric] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    # Row-major cell sizes: [{"w": 1.0, "h": 1.0}, ...]
    cell_sizes: list[dict[str, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Computed properties                                                  #
    # ------------------------------------------------------------------ #

    def _cell_index(self, x: int, y: int) -> int:
        return y * self.grid_width + x

    def cell_size_at(self, x: int, y: int) -> tuple[float, float]:
        idx = self._cell_index(x, y)
        if idx < 0 or idx >= len(self.cell_sizes):
            return (0.0, 0.0)
        entry = self.cell_sizes[idx]
        return (float(entry.get("w", 0.0)), float(entry.get("h", 0.0)))

    def column_widths(self) -> list[float]:
        widths: list[float] = []
        for x in range(self.grid_width):
            w, _ = self.cell_size_at(x, 0)
            widths.append(w)
        return widths

    def row_heights(self) -> list[float]:
        heights: list[float] = []
        for y in range(self.grid_height):
            _, h = self.cell_size_at(0, y)
            heights.append(h)
        return heights

    def block_dimensions_in(self, block: Block) -> tuple[float, float]:
        col_widths = self.column_widths()
        row_heights = self.row_heights()
        w_in = sum(col_widths[block.x:block.x + block.width])
        h_in = sum(row_heights[block.y:block.y + block.height])
        return (round(w_in, 4), round(h_in, 4))

    @property
    def finished_width_in(self) -> float:
        return round(sum(self.column_widths()), 4)

    @property
    def finished_height_in(self) -> float:
        return round(sum(self.row_heights()), 4)

    @property
    def fabric_map(self) -> dict[str, Fabric]:
        return {f.id: f for f in self.fabrics}

    # ------------------------------------------------------------------ #
    # Grid helpers                                                         #
    # ------------------------------------------------------------------ #

    def cell_grid(self) -> dict[tuple[int, int], str]:
        """Return a dict mapping (x, y) -> fabric_id for every assigned cell."""
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
        expected_cells = self.grid_width * self.grid_height

        if len(self.cell_sizes) != expected_cells:
            errors.append(
                f"cell_sizes length {len(self.cell_sizes)} != grid cells {expected_cells}"
            )

        # Enforce row/column consistency to guarantee tiling
        for x in range(self.grid_width):
            base_w, _ = self.cell_size_at(x, 0)
            for y in range(1, self.grid_height):
                w, _ = self.cell_size_at(x, y)
                if abs(w - base_w) > 1e-6:
                    errors.append(
                        f"Column {x} width mismatch at row {y}: {w} vs {base_w}"
                    )
                    break
        for y in range(self.grid_height):
            _, base_h = self.cell_size_at(0, y)
            for x in range(1, self.grid_width):
                _, h = self.cell_size_at(x, y)
                if abs(h - base_h) > 1e-6:
                    errors.append(
                        f"Row {y} height mismatch at col {x}: {h} vs {base_h}"
                    )
                    break

        if self.quilt_width_in > 0 and abs(self.finished_width_in - self.quilt_width_in) > 1e-4:
            errors.append(
                f"Finished width {self.finished_width_in} != quilt_width_in {self.quilt_width_in}"
            )
        if self.quilt_height_in > 0 and abs(self.finished_height_in - self.quilt_height_in) > 1e-4:
            errors.append(
                f"Finished height {self.finished_height_in} != quilt_height_in {self.quilt_height_in}"
            )

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
            for corner_name, corner_fab in (block.corners or {}).items():
                if corner_fab not in fabric_ids:
                    errors.append(
                        f"Block {i} corner '{corner_name}' references unknown fabric_id='{corner_fab}'"
                    )
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
        area: dict[str, float] = {f.id: 0.0 for f in self.fabrics}
        for block in self.blocks:
            for (cx, cy) in block.cells():
                w, h = self.cell_size_at(cx, cy)
                cell_area = w * h
                base_area = cell_area
                if block.corners:
                    # If a corner triangle exists in this cell, split area 50/50
                    corner_keys = set(block.corners.keys())
                    is_nw = (cx == block.x and cy == block.y)
                    is_ne = (cx == block.x + block.width - 1 and cy == block.y)
                    is_sw = (cx == block.x and cy == block.y + block.height - 1)
                    is_se = (cx == block.x + block.width - 1 and cy == block.y + block.height - 1)
                    corner_map = {"nw": is_nw, "ne": is_ne, "sw": is_sw, "se": is_se}
                    for name, is_corner_cell in corner_map.items():
                        if is_corner_cell and name in corner_keys:
                            base_area = cell_area * 0.5
                            area[block.corners[name]] = area.get(block.corners[name], 0.0) + (cell_area * 0.5)
                area[block.fabric_id] = area.get(block.fabric_id, 0.0) + base_area
        for fabric in self.fabrics:
            fabric.total_sqin = round(area[fabric.id], 2)

    # ------------------------------------------------------------------ #
    # Cutting chart                                                        #
    # ------------------------------------------------------------------ #

    def to_cutting_chart(self) -> CuttingChart:
        """
        Group blocks by fabric_id + (width, height) -> CutPiece entries.
        Each CutPiece represents one type of rectangular cut for one fabric.
        """
        self.compute_fabric_areas()
        fabric_map = self.fabric_map

        # key: (fabric_id, cut_width_in, cut_height_in, piece_type) -> count
        piece_counts: dict[tuple[str, float, float, str], int] = {}

        for block in self.blocks:
            w_in, h_in = self.block_dimensions_in(block)
            w_in = round(w_in + 2 * self.seam_allowance, 4)
            h_in = round(h_in + 2 * self.seam_allowance, 4)
            # Normalize so width <= height for consistent grouping
            if w_in > h_in:
                w_in, h_in = h_in, w_in
            key = (block.fabric_id, w_in, h_in, "base")
            piece_counts[key] = piece_counts.get(key, 0) + 1

            # Corner squares: full-cell stitch-and-flip squares for corners
            for corner_name, corner_fab in (block.corners or {}).items():
                if corner_name == "nw":
                    cx, cy = block.x, block.y
                elif corner_name == "ne":
                    cx, cy = block.x + block.width - 1, block.y
                elif corner_name == "sw":
                    cx, cy = block.x, block.y + block.height - 1
                else:
                    cx, cy = block.x + block.width - 1, block.y + block.height - 1
                cw, ch = self.cell_size_at(cx, cy)
                square = round(max(cw, ch) + 2 * self.seam_allowance, 4)
                key = (corner_fab, square, square, "corner")
                piece_counts[key] = piece_counts.get(key, 0) + 1

        chart = CuttingChart(
            block_size_in=0.0,
            seam_allowance=self.seam_allowance,
        )
        for (fabric_id, w_in, h_in, piece_type), qty in sorted(piece_counts.items()):
            fab = fabric_map.get(fabric_id)
            chart.pieces.append(CutPiece(
                fabric_id=fabric_id,
                fabric_name=fab.name if fab else fabric_id,
                color_hex=fab.color_hex if fab else "#888888",
                cut_width_in=w_in,
                cut_height_in=h_in,
                quantity=qty,
                piece_type=piece_type,
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
            "quilt_width_in": self.quilt_width_in,
            "quilt_height_in": self.quilt_height_in,
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
                 "fabric_id": b.fabric_id, "corners": b.corners or {}}
                for b in self.blocks
            ],
            "cell_sizes": self.cell_sizes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "QuiltPattern":
        pattern = cls(
            grid_width=data["grid_width"],
            grid_height=data["grid_height"],
            quilt_width_in=data.get("quilt_width_in", 0.0),
            quilt_height_in=data.get("quilt_height_in", 0.0),
            seam_allowance=data.get("seam_allowance", 0.25),
        )
        pattern.cell_sizes = data.get("cell_sizes", [])
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
                corners=b.get("corners") or {},
            ))
        return pattern
