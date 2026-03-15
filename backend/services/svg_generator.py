"""
StarVector 8B — text-to-SVG generation using StarVector-8B.

Generates native SVG from text prompts. SVG output contains solid-fill
geometric shapes that map naturally to quilt blocks, bypassing lossy
raster-to-grid conversion.

Follows flux_pipeline.py patterns:
  - Module-level lazy state
  - Silent failure with logging
  - VRAM management (unload to free for FLUX fallback)

Requires (already in requirements.txt):
  transformers, torch

The StarVector model (starvector-8b) must be downloaded separately:
  Model: joanrodai/starvector-8b (HuggingFace)
"""
from __future__ import annotations

import io
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Env-configurable settings
STARVECTOR_MODEL_ID = os.environ.get(
    "STARVECTOR_MODEL_ID", "joanrodai/starvector-8b"
)
STARVECTOR_MAX_LENGTH = int(os.environ.get("STARVECTOR_MAX_LENGTH", "4096"))

# Style suffix appended to prompts for quilt-appropriate SVG
STYLE_SUFFIX = (
    ", geometric quilt pattern with solid-fill rectangles and squares, "
    "bold flat colors, clean grid layout, no gradients, no textures"
)

# Module-level state
_backend: str = "none"  # "cuda", "cpu", "none"
_model = None
_processor = None


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model() -> None:
    """Load StarVector model. Tries CUDA first, logs and returns silently on failure."""
    global _model, _processor, _backend

    if _backend != "none":
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else None
        if device is None:
            logger.info("CUDA not available for StarVector, skipping")
            return

        logger.info(f"Loading StarVector model: {STARVECTOR_MODEL_ID}")

        model = AutoModelForCausalLM.from_pretrained(
            STARVECTOR_MODEL_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.to(device)
        model.eval()

        processor = AutoProcessor.from_pretrained(
            STARVECTOR_MODEL_ID,
            trust_remote_code=True,
        )

        _model = model
        _processor = processor
        _backend = "cuda"
        logger.info("StarVector loaded on CUDA")

    except ImportError as e:
        logger.info(f"StarVector unavailable (import): {e}")
    except Exception as e:
        logger.warning(f"StarVector loading failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_quilt_svg(
    prompt: str,
    grid_width: int = 40,
    grid_height: int = 50,
) -> Optional[str]:
    """
    Generate SVG from a text prompt using StarVector.

    Tries three strategies in order:
      1. generate_text2svg() if model has this method
      2. Base generate() with SVG-instructing prompt prefix
      3. Generate reference grid image via Pillow, then im2svg

    Returns SVG string or None if generation fails.
    """
    _load_model()

    if _backend == "none" or _model is None:
        return None

    full_prompt = prompt + STYLE_SUFFIX
    svg = None

    # Strategy 1: text2svg method
    svg = _try_text2svg(full_prompt)
    if svg:
        return svg

    # Strategy 2: generate with SVG prompt prefix
    svg = _try_generate_with_prefix(full_prompt)
    if svg:
        return svg

    # Strategy 3: reference grid image → im2svg
    svg = _try_im2svg(full_prompt, grid_width, grid_height)
    if svg:
        return svg

    logger.warning("All StarVector generation strategies failed")
    return None


def _try_text2svg(prompt: str) -> Optional[str]:
    """Strategy 1: Use model's text2svg method if available."""
    try:
        if not hasattr(_model, "generate_text2svg"):
            return None

        result = _model.generate_text2svg(prompt, max_length=STARVECTOR_MAX_LENGTH)
        svg = _extract_svg(result if isinstance(result, str) else str(result))
        if svg:
            logger.info("StarVector: text2svg succeeded")
        return svg
    except Exception as e:
        logger.info(f"StarVector text2svg failed: {e}")
        return None


def _try_generate_with_prefix(prompt: str) -> Optional[str]:
    """Strategy 2: Use base generate() with SVG-instructing prompt."""
    try:
        import torch

        svg_prompt = f"Generate an SVG image: {prompt}"
        inputs = _processor(svg_prompt, return_tensors="pt")

        # Move inputs to model device
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=STARVECTOR_MAX_LENGTH,
                do_sample=True,
                temperature=0.7,
            )

        decoded = _processor.batch_decode(outputs, skip_special_tokens=True)
        raw = decoded[0] if decoded else ""
        svg = _extract_svg(raw)
        if svg:
            logger.info("StarVector: generate with prefix succeeded")
        return svg
    except Exception as e:
        logger.info(f"StarVector generate-with-prefix failed: {e}")
        return None


def _try_im2svg(prompt: str, grid_width: int, grid_height: int) -> Optional[str]:
    """
    Strategy 3: Generate a simple reference grid image via Pillow,
    then use model's im2svg capability (StarVector's primary mode).
    """
    try:
        from PIL import Image

        # Create a simple colored grid as reference
        img_w, img_h = 384, 384
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))

        # Draw some colored rectangles as reference
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        colors = [
            (27, 45, 91), (196, 52, 40), (245, 240, 220),
            (74, 124, 63), (212, 164, 42), (125, 184, 216),
        ]
        cell_w = img_w / min(grid_width, 8)
        cell_h = img_h / min(grid_height, 8)
        for gy in range(min(grid_height, 8)):
            for gx in range(min(grid_width, 8)):
                color = colors[(gx + gy) % len(colors)]
                x0 = int(gx * cell_w)
                y0 = int(gy * cell_h)
                x1 = int((gx + 1) * cell_w)
                y1 = int((gy + 1) * cell_h)
                draw.rectangle([x0, y0, x1, y1], fill=color)

        # Use im2svg
        if hasattr(_model, "generate_im2svg"):
            result = _model.generate_im2svg(
                img, max_length=STARVECTOR_MAX_LENGTH
            )
            svg = _extract_svg(result if isinstance(result, str) else str(result))
            if svg:
                logger.info("StarVector: im2svg succeeded")
            return svg

        # Alternative: pass image through processor
        import torch
        inputs = _processor(images=img, return_tensors="pt")
        device = next(_model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=STARVECTOR_MAX_LENGTH,
                do_sample=False,
            )

        decoded = _processor.batch_decode(outputs, skip_special_tokens=True)
        raw = decoded[0] if decoded else ""
        svg = _extract_svg(raw)
        if svg:
            logger.info("StarVector: im2svg (processor) succeeded")
        return svg

    except Exception as e:
        logger.info(f"StarVector im2svg failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_svg(raw: str) -> Optional[str]:
    """
    Extract and validate SVG content from model output.
    Strips content outside <svg>...</svg>, validates basic structure.
    """
    if not raw:
        return None

    # Find <svg ...>...</svg>
    match = re.search(r"(<svg[^>]*>.*?</svg>)", raw, re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    svg = match.group(1)

    # Basic validation: must contain at least one shape element
    shape_tags = ("rect", "polygon", "path", "circle", "ellipse", "line", "polyline")
    has_shape = any(f"<{tag}" in svg.lower() for tag in shape_tags)
    if not has_shape:
        logger.info("SVG has no shape elements, discarding")
        return None

    return svg


# ─────────────────────────────────────────────────────────────────────────────
# VRAM management
# ─────────────────────────────────────────────────────────────────────────────

def unload() -> None:
    """Unload model and free VRAM so FLUX can load."""
    global _model, _processor, _backend

    if _model is not None:
        del _model
        _model = None
    if _processor is not None:
        del _processor
        _processor = None

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    _backend = "none"
    logger.info("StarVector unloaded, VRAM freed")


def generator_status() -> dict:
    """Return current StarVector status."""
    return {
        "loaded": _backend != "none",
        "type": _backend,
        "model_id": STARVECTOR_MODEL_ID,
    }
