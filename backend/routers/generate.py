"""POST /api/generate — text prompt → quilt pattern."""
from __future__ import annotations

import base64
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import GenerateRequest
from ..services import (
    flux_pipeline,
    grid_extractor,
    grid_engine,
    svg_renderer,
    cutting_calculator,
    ollama_client,
)

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)


@router.post("/generate")
async def generate_pattern(req: GenerateRequest) -> dict[str, Any]:
    """
    Generate a quilt pattern from a text prompt.

    Flow:
      1. FLUX.1-dev generates a quilt-style image (or skip if unavailable)
      2. Grid extractor converts image → QuiltPattern
      3. Grid engine validates + computes cutting chart
      4. Ollama writes a prose guide
      5. SVG renderer produces a preview
    """
    # Step 1: Generate image
    image_bytes: bytes | None = None
    try:
        image_bytes = flux_pipeline.generate_quilt_image(
            prompt=req.prompt,
            width=1024,
            height=1024,
        )
    except Exception as e:
        logger.warning(f"Image generation failed: {e}, proceeding without image")

    # Step 2: Extract grid (from image or synthetic fallback)
    confidence = 0.0
    if image_bytes:
        pattern, confidence = grid_extractor.extract_pattern_from_image(
            image_bytes=image_bytes,
            grid_width=req.grid_width,
            grid_height=req.grid_height,
            palette_size=req.palette_size,
            block_size_in=req.block_size_inches,
        )
    else:
        # Try Ollama layout generation as intermediate fallback
        try:
            layout_json = await ollama_client.generate_block_layout(
                prompt=req.prompt,
                grid_width=req.grid_width,
                grid_height=req.grid_height,
                palette_size=req.palette_size,
            )
            if layout_json:
                pattern = grid_engine.QuiltPattern.from_dict({
                    "grid_width": req.grid_width,
                    "grid_height": req.grid_height,
                    "block_size_in": req.block_size_inches,
                    "seam_allowance": 0.25,
                    **layout_json,
                })
                validation_errors = pattern.validate()
                if not validation_errors:
                    confidence = 0.5
                    logger.info("Using Ollama-generated block layout")
                else:
                    logger.warning(f"Ollama layout failed validation: {validation_errors}")
                    raise ValueError("Ollama layout failed validation")
            else:
                raise ValueError("Ollama returned empty layout")
        except Exception as e:
            logger.warning(f"Ollama layout fallback failed: {e}, using synthetic stripes")
            pattern = grid_extractor._synthetic_fallback(
                grid_width=req.grid_width,
                grid_height=req.grid_height,
                palette_size=req.palette_size,
                block_size_in=req.block_size_inches,
                seam_allowance=0.25,
            )
            confidence = 0.0

    # Step 3: Validate
    errors = pattern.validate()
    if errors:
        logger.warning(f"Pattern validation errors: {errors}")

    # Step 4: Cutting chart
    chart = pattern.to_cutting_chart()
    cut_instructions = cutting_calculator.format_cutting_sequence(chart, pattern.fabrics)

    # Step 5: Guide (Ollama)
    guide_text = ""
    try:
        pattern_dict = pattern.to_dict()
        guide_text = await ollama_client.generate_guide(
            pattern_json=pattern_dict,
            cutting_instructions=cut_instructions,
            title=f"Quilt: {req.prompt[:50]}",
        )
    except Exception as e:
        logger.warning(f"Guide generation failed: {e}")
        guide_text = "\n".join(cut_instructions)

    # Step 6: SVG
    grid_svg = svg_renderer.render_grid_svg(pattern)
    cutting_svg = svg_renderer.render_cutting_diagram_svg(chart, pattern)

    # Build cutting chart JSON
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
        "pipeline_status": flux_pipeline.pipeline_status(),
        "image_b64": base64.b64encode(image_bytes).decode() if image_bytes else None,
    }
