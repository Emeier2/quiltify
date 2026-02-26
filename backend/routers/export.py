"""GET /api/export — download pattern as SVG, PDF, or CSV."""
from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from ..models.requests import GuideRequest
from ..services import grid_engine, cutting_calculator, svg_renderer

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/export/svg")
async def export_svg(req: GuideRequest) -> Response:
    pattern = grid_engine.QuiltPattern.from_dict(req.pattern.model_dump())
    svg = svg_renderer.render_grid_svg(pattern, cell_px=16)
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Content-Disposition": 'attachment; filename="quilt-pattern.svg"'},
    )


@router.post("/export/csv")
async def export_csv(req: GuideRequest) -> StreamingResponse:
    pattern = grid_engine.QuiltPattern.from_dict(req.pattern.model_dump())
    chart = pattern.to_cutting_chart()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Fabric", "Color Hex", "Cut Width (in)", "Cut Height (in)", "Quantity"])
    for piece in chart.pieces:
        writer.writerow([
            piece.fabric_name,
            piece.color_hex,
            piece.cut_width_in,
            piece.cut_height_in,
            piece.quantity,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cutting-chart.csv"'},
    )


@router.post("/export/pdf")
async def export_pdf(req: GuideRequest) -> Response:
    """Export as PDF using weasyprint if available, otherwise return error."""
    try:
        import weasyprint
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires weasyprint. Install with: pip install weasyprint"
        )

    pattern = grid_engine.QuiltPattern.from_dict(req.pattern.model_dump())
    chart = pattern.to_cutting_chart()
    svg = svg_renderer.render_grid_svg(pattern, cell_px=10)
    cut_instructions = cutting_calculator.format_cutting_sequence(chart, pattern.fabrics)

    html = _build_pdf_html(pattern, chart, svg, cut_instructions)
    pdf_bytes = weasyprint.HTML(string=html).write_pdf()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="quilt-guide.pdf"'},
    )


def _build_pdf_html(
    pattern: grid_engine.QuiltPattern,
    chart: grid_engine.CuttingChart,
    svg: str,
    cut_instructions: list[str],
) -> str:
    table_rows = "".join(
        f"<tr><td>{p.fabric_name}</td><td>{p.color_hex}</td>"
        f"<td>{p.cut_width_in}\"</td><td>{p.cut_height_in}\"</td><td>{p.quantity}</td></tr>"
        for p in chart.pieces
    )
    instructions_html = "<br>".join(cut_instructions)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: Georgia, serif; max-width: 800px; margin: 40px auto; color: #333; }}
  h1 {{ font-size: 28px; border-bottom: 2px solid #333; }}
  h2 {{ font-size: 20px; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #f0ede8; }}
  .quilt-svg {{ max-width: 500px; margin: 20px auto; display: block; }}
</style>
</head>
<body>
<h1>Quiltify — Pattern Guide</h1>
<p>Finished size: {pattern.finished_width_in:.1f}" × {pattern.finished_height_in:.1f}"
   &nbsp;|&nbsp; Block size: {pattern.block_size_in}" finished
   &nbsp;|&nbsp; Seam allowance: {pattern.seam_allowance}"</p>

<div class="quilt-svg">{svg}</div>

<h2>Cutting Chart</h2>
<table>
  <tr><th>Fabric</th><th>Color</th><th>Width</th><th>Height</th><th>Qty</th></tr>
  {table_rows}
</table>

<h2>Cutting Instructions</h2>
<p>{instructions_html}</p>
</body>
</html>"""
