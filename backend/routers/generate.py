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
    svg_generator,
    svg_pattern_parser,
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
      Phase 1: Try StarVector SVG generation (direct SVG → QuiltPattern)
      Phase 2: Fallback to FLUX raster pipeline + Ollama layout + synthetic
      Phase 3: Validate, cutting chart, guide, SVG render
    """
    pattern = None
    confidence = 0.0
    source = "starvector"
    image_bytes: bytes | None = None

    # ── Phase 1: Try StarVector ──────────────────────────────────────────
    try:
        svg_str = svg_generator.generate_quilt_svg(
            prompt=req.prompt,
            grid_width=req.grid_width,
            grid_height=req.grid_height,
        )
        if svg_str:
            sv_pattern, sv_confidence = svg_pattern_parser.parse_svg_to_pattern(
                svg_str=svg_str,
                grid_width=req.grid_width,
                grid_height=req.grid_height,
                palette_size=req.palette_size,
                quilt_width_in=req.quilt_width_in,
                quilt_height_in=req.quilt_height_in,
            )
            sv_errors = sv_pattern.validate()
            if not sv_errors:
                pattern = sv_pattern
                confidence = sv_confidence
                source = "starvector"
                logger.info(f"StarVector pipeline succeeded (confidence={confidence})")
            else:
                logger.warning(f"StarVector pattern invalid, falling back: {sv_errors}")
    except Exception as e:
        logger.warning(f"StarVector pipeline failed, falling back: {e}")

    # ── Phase 2: Fallback to FLUX / Ollama / synthetic ───────────────────
    if pattern is None:
        # Free StarVector VRAM so FLUX can load
        svg_generator.unload()
        source = "flux"

        # Step 2a: Generate image via FLUX
        try:
            image_bytes = flux_pipeline.generate_quilt_image(
                prompt=req.prompt,
                width=1024,
                height=1024,
            )
        except Exception as e:
            logger.warning(f"Image generation failed: {e}, proceeding without image")

        # Step 2b: Extract grid (from image or fallback)
        if image_bytes:
            pattern, confidence = grid_extractor.extract_pattern_from_image(
                image_bytes=image_bytes,
                grid_width=req.grid_width,
                grid_height=req.grid_height,
                palette_size=req.palette_size,
                quilt_width_in=req.quilt_width_in,
                quilt_height_in=req.quilt_height_in,
            )
        else:
            # Try Ollama layout generation as intermediate fallback
            try:
                layout_json = await ollama_client.generate_block_layout(
                    prompt=req.prompt,
                    grid_width=req.grid_width,
                    grid_height=req.grid_height,
                    palette_size=req.palette_size,
                    quilt_width_in=req.quilt_width_in,
                    quilt_height_in=req.quilt_height_in,
                )
                if layout_json:
                    pattern = grid_engine.QuiltPattern.from_dict({
                        "grid_width": req.grid_width,
                        "grid_height": req.grid_height,
                        "quilt_width_in": req.quilt_width_in,
                        "quilt_height_in": req.quilt_height_in,
                        "seam_allowance": 0.25,
                        **layout_json,
                    })
                    validation_errors = pattern.validate()
                    if not validation_errors:
                        confidence = 0.5
                        source = "ollama"
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
                    quilt_width_in=req.quilt_width_in,
                    quilt_height_in=req.quilt_height_in,
                    seam_allowance=0.25,
                )
                confidence = 0.0
                source = "synthetic"

    # ── Phase 3: Validate, chart, guide, SVG ─────────────────────────────
    errors = pattern.validate()
    if errors:
        logger.warning(f"Pattern validation errors: {errors}")

    # Cutting chart
    chart = pattern.to_cutting_chart()
    cut_instructions = cutting_calculator.format_cutting_sequence(chart, pattern.fabrics)

    # Guide (Ollama)
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

    # SVG
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
            "piece_type": p.piece_type,
        }
        for p in chart.pieces
    ]

    # Pipeline status
    flux_status = flux_pipeline.pipeline_status()
    starvector_status = svg_generator.generator_status()

    return {
        "pattern_json": pattern.to_dict(),
        "svg": grid_svg,
        "cutting_svg": cutting_svg,
        "cutting_chart": cutting_chart_json,
        "guide": guide_text,
        "confidence_score": confidence,
        "validation_errors": errors,
        "pipeline_status": {
            **flux_status,
            "starvector": starvector_status,
            "source": source,
        },
        "image_b64": base64.b64encode(image_bytes).decode() if image_bytes else None,
    }
