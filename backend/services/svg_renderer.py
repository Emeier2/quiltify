"""
SVG Renderer — produces grid preview SVGs from a QuiltPattern.

Outputs:
  - Full-grid color preview (one colored rectangle per block)
  - Cutting diagram (one labeled rectangle per CutPiece type, per fabric)
"""
from __future__ import annotations

import io
from typing import Optional

try:
    import svgwrite
    _HAS_SVGWRITE = True
except ImportError:
    _HAS_SVGWRITE = False

from .grid_engine import QuiltPattern, CuttingChart


# Pixels per grid-unit in the SVG preview
CELL_PX = 12


def render_grid_svg(pattern: QuiltPattern, cell_px: int = CELL_PX) -> str:
    """Return SVG string: colored grid blocks with thin stroke lines."""
    if not _HAS_SVGWRITE:
        return _fallback_svg(pattern, cell_px)

    width_px = pattern.grid_width * cell_px
    height_px = pattern.grid_height * cell_px

    dwg = svgwrite.Drawing(size=(f"{width_px}px", f"{height_px}px"),
                           profile="full")
    dwg.viewbox(0, 0, width_px, height_px)

    fabric_map = pattern.fabric_map

    # Background
    dwg.add(dwg.rect(insert=(0, 0), size=(width_px, height_px), fill="#f5f5f0"))

    for block in pattern.blocks:
        fab = fabric_map.get(block.fabric_id)
        color = fab.color_hex if fab else "#cccccc"
        x = block.x * cell_px
        y = block.y * cell_px
        w = block.width * cell_px
        h = block.height * cell_px
        dwg.add(dwg.rect(
            insert=(x, y), size=(w, h),
            fill=color,
            stroke="#ffffff",
            stroke_width=0.5,
        ))

    return dwg.tostring()


def render_cutting_diagram_svg(
    chart: CuttingChart,
    pattern: QuiltPattern,
    max_width_px: int = 800,
) -> str:
    """
    Render a cutting diagram: for each fabric, show a scaled representation
    of each cut piece type with dimensions labeled.
    """
    if not _HAS_SVGWRITE:
        return "<svg xmlns='http://www.w3.org/2000/svg'><text y='20'>Install svgwrite</text></svg>"

    by_fabric = chart.by_fabric()
    fabric_map = pattern.fabric_map

    # Layout: stack fabric sections vertically
    SCALE = 20  # 1 inch = 20px
    PADDING = 30
    PIECE_GAP = 20
    SECTION_GAP = 40
    LABEL_HEIGHT = 24

    sections: list[dict] = []
    total_height = PADDING

    for fabric_id, pieces in by_fabric.items():
        fab = fabric_map.get(fabric_id)
        color = fab.color_hex if fab else "#cccccc"
        name = fab.name if fab else fabric_id

        section_pieces = []
        row_x = PADDING
        row_height = 0
        section_height = LABEL_HEIGHT + 10

        for piece in sorted(pieces, key=lambda p: p.cut_width_in * p.cut_height_in, reverse=True):
            w_px = round(piece.cut_width_in * SCALE)
            h_px = round(piece.cut_height_in * SCALE)
            if row_x + w_px + PADDING > max_width_px:
                row_x = PADDING
                section_height += row_height + PIECE_GAP
                row_height = 0
            section_pieces.append({
                "piece": piece,
                "x": row_x,
                "y_offset": section_height,
                "w_px": w_px,
                "h_px": h_px,
            })
            row_x += w_px + PIECE_GAP
            row_height = max(row_height, h_px + LABEL_HEIGHT)

        section_height += row_height + PADDING
        sections.append({
            "fabric_id": fabric_id,
            "color": color,
            "name": name,
            "y_start": total_height,
            "height": section_height,
            "pieces": section_pieces,
        })
        total_height += section_height + SECTION_GAP

    dwg = svgwrite.Drawing(size=(f"{max_width_px}px", f"{total_height}px"), profile="full")
    dwg.viewbox(0, 0, max_width_px, total_height)
    dwg.add(dwg.rect(insert=(0, 0), size=(max_width_px, total_height), fill="#fafaf8"))

    for section in sections:
        sy = section["y_start"]
        # Section label
        dwg.add(dwg.text(
            section["name"],
            insert=(PADDING, sy + 18),
            font_size="14px",
            font_family="sans-serif",
            font_weight="bold",
            fill="#333",
        ))

        for item in section["pieces"]:
            piece = item["piece"]
            px = item["x"]
            py = sy + item["y_offset"]
            pw = item["w_px"]
            ph = item["h_px"]

            # Piece rectangle
            dwg.add(dwg.rect(
                insert=(px, py), size=(pw, ph),
                fill=section["color"],
                stroke="#555",
                stroke_width=1,
            ))
            # Dimensions label inside
            label = f'{piece.cut_width_in}" × {piece.cut_height_in}"'
            qty_label = f"×{piece.quantity}"
            dwg.add(dwg.text(
                label,
                insert=(px + 4, py + min(14, ph - 4)),
                font_size="10px",
                font_family="sans-serif",
                fill=_contrasting_text(section["color"]),
            ))
            if ph > 28:
                dwg.add(dwg.text(
                    qty_label,
                    insert=(px + 4, py + 26),
                    font_size="11px",
                    font_family="sans-serif",
                    font_weight="bold",
                    fill=_contrasting_text(section["color"]),
                ))

    return dwg.tostring()


def _contrasting_text(hex_color: str) -> str:
    """Return '#000' or '#fff' depending on background luminance."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) < 6:
        return "#000"
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000" if luminance > 0.5 else "#fff"


def _fallback_svg(pattern: QuiltPattern, cell_px: int) -> str:
    """Minimal SVG fallback when svgwrite is not installed."""
    width_px = pattern.grid_width * cell_px
    height_px = pattern.grid_height * cell_px
    fabric_map = pattern.fabric_map

    rects = []
    for block in pattern.blocks:
        fab = fabric_map.get(block.fabric_id)
        color = fab.color_hex if fab else "#cccccc"
        x = block.x * cell_px
        y = block.y * cell_px
        w = block.width * cell_px
        h = block.height * cell_px
        rects.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{color}" stroke="#fff" stroke-width="0.5"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
        f'<rect width="{width_px}" height="{height_px}" fill="#f5f5f0"/>'
        + "".join(rects)
        + "</svg>"
    )
