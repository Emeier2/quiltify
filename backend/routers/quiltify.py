"""POST /api/quiltify — input image → quilt pattern."""
from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import QuiltifyRequest
from ..services import (
    quiltification,
    grid_extractor,
    svg_renderer,
    cutting_calculator,
    ollama_client,
    flux_pipeline,
)

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/quiltify")
async def quiltify_image(req: QuiltifyRequest) -> dict[str, Any]:
    """
    Transform an input image into a quilt pattern.

    Flow:
      1. Decode base64 input image
      2. SAM + ControlNet img2img → quilt-style image
      3. Grid extractor → QuiltPattern
      4. Grid engine validates + computes cutting chart
      5. Ollama writes guide
      6. SVG renderer produces preview
    """
    # Step 1: Decode input image
    try:
        original_bytes = grid_extractor.image_bytes_from_base64(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    # Step 2: Quiltify (SAM + ControlNet)
    quilt_image_bytes: bytes | None = None
    try:
        quilt_image_bytes = quiltification.quiltify_image(
            image_bytes=original_bytes,
            prompt="modern geometric quilt",
        )
    except Exception as e:
        logger.warning(f"Quiltification failed: {e}, using original image for extraction")
        quilt_image_bytes = original_bytes

    # Step 3: Extract grid
    source_bytes = quilt_image_bytes or original_bytes
    pattern, confidence = grid_extractor.extract_pattern_from_image(
        image_bytes=source_bytes,
        grid_width=req.grid_width,
        grid_height=req.grid_height,
        palette_size=req.palette_size,
        block_size_in=req.block_size_inches,
    )

    # Step 4: Validate + cutting chart
    errors = pattern.validate()
    chart = pattern.to_cutting_chart()
    cut_instructions = cutting_calculator.format_cutting_sequence(chart, pattern.fabrics)

    # Step 5: Guide
    guide_text = ""
    try:
        guide_text = await ollama_client.generate_guide(
            pattern_json=pattern.to_dict(),
            cutting_instructions=cut_instructions,
            title="Quiltified Image Pattern",
        )
    except Exception as e:
        logger.warning(f"Guide generation failed: {e}")
        guide_text = "\n".join(cut_instructions)

    # Step 6: SVG
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
        "pattern_json": pattern.to_dict(),
        "svg": grid_svg,
        "cutting_svg": cutting_svg,
        "cutting_chart": cutting_chart_json,
        "guide": guide_text,
        "confidence_score": confidence,
        "validation_errors": errors,
        "original_image_b64": base64.b64encode(original_bytes).decode(),
        "quilt_image_b64": base64.b64encode(quilt_image_bytes).decode() if quilt_image_bytes else None,
    }
