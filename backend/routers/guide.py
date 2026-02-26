"""POST /api/guide — re-generate guide from an edited pattern."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import GuideRequest
from ..services import grid_engine, cutting_calculator, ollama_client, svg_renderer

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/guide")
async def regenerate_guide(req: GuideRequest) -> dict[str, Any]:
    """
    Re-generate a quilting guide from an (optionally edited) QuiltPattern.
    Called when the user edits the grid in the frontend.
    """
    pattern = grid_engine.QuiltPattern.from_dict(req.pattern.model_dump())

    errors = pattern.validate()
    if errors:
        logger.warning(f"Guide request pattern errors: {errors}")

    chart = pattern.to_cutting_chart()
    cut_instructions = cutting_calculator.format_cutting_sequence(chart, pattern.fabrics)

    guide_text = ""
    try:
        guide_text = await ollama_client.generate_guide(
            pattern_json=pattern.to_dict(),
            cutting_instructions=cut_instructions,
            title=req.title,
        )
    except Exception as e:
        logger.warning(f"Guide generation failed: {e}")
        guide_text = "\n".join(cut_instructions)

    grid_svg = svg_renderer.render_grid_svg(pattern)
    cutting_svg = svg_renderer.render_cutting_diagram_svg(chart, pattern)

    cutting_chart_json = [
        {
            "fabric_id": p.fabric_id,
            "fabric_name": p.fabric_name,
            "color_hex": p.color_hex,
            "cut_width_in": p.cut_width_in,
            "cut_height_in": p.cut_height_in,
            "quantity": p.quantity,
        }
        for p in chart.pieces
    ]

    return {
        "guide": guide_text,
        "cutting_chart": cutting_chart_json,
        "svg": grid_svg,
        "cutting_svg": cutting_svg,
        "pattern_json": pattern.to_dict(),
        "validation_errors": errors,
    }
