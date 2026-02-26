"""
FLUX Pipeline — text-to-image generation using FLUX.1-dev.

Loads FLUX.1-dev in Q4 quantization via diffusers + bitsandbytes.
On first call the model is loaded and cached for subsequent requests.

Requires:
  pip install diffusers transformers accelerate bitsandbytes torch
  A HuggingFace token with access to black-forest-labs/FLUX.1-dev
"""
from __future__ import annotations

import io
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Model IDs
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_MODEL_ID_SCHNELL = "black-forest-labs/FLUX.1-schnell"  # fallback, no auth needed

# Elizabeth Hartman style suffix appended to all prompts
HARTMAN_SUFFIX = (
    ", Elizabeth Hartman modern quilt pattern, solid fabric geometric squares and rectangles, "
    "bold solid colors, pictorial patchwork design, clean grid lines, quilt top view, "
    "no gradients no shading no textures"
)

_pipeline = None
_pipeline_type: str = "none"


def _load_pipeline() -> None:
    global _pipeline, _pipeline_type

    if _pipeline is not None:
        return

    try:
        import torch
        from diffusers import FluxPipeline, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN")

        # Try Q4 FLUX.1-dev first
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        try:
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID,
                quantization_config=nf4_config,
                torch_dtype=torch.float16,
                token=hf_token,
            )
            pipe.enable_model_cpu_offload()
            _pipeline = pipe
            _pipeline_type = "flux-dev-q4"
            logger.info("Loaded FLUX.1-dev (Q4)")
            return
        except Exception as e:
            logger.warning(f"FLUX.1-dev load failed ({e}), trying schnell...")

        # Fallback: FLUX.1-schnell (no auth required, faster, slightly lower quality)
        try:
            pipe = FluxPipeline.from_pretrained(
                FLUX_MODEL_ID_SCHNELL,
                torch_dtype=torch.bfloat16,
            )
            pipe.enable_model_cpu_offload()
            _pipeline = pipe
            _pipeline_type = "flux-schnell"
            logger.info("Loaded FLUX.1-schnell (fallback)")
            return
        except Exception as e:
            logger.warning(f"FLUX.1-schnell load failed ({e})")

    except ImportError as e:
        logger.warning(f"diffusers/torch not available: {e}")

    _pipeline_type = "none"


def generate_quilt_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
) -> Optional[bytes]:
    """
    Generate a quilt-style image from a text prompt.
    Returns JPEG bytes, or None if the pipeline is unavailable.
    """
    _load_pipeline()

    if _pipeline is None:
        logger.info("No image generation pipeline available, returning None")
        return None

    import torch

    full_prompt = prompt + HARTMAN_SUFFIX
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    kwargs: dict = {
        "prompt": full_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    if generator is not None:
        kwargs["generator"] = generator

    result = _pipeline(**kwargs)
    pil_image = result.images[0]

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def pipeline_status() -> dict:
    return {
        "loaded": _pipeline is not None,
        "type": _pipeline_type,
    }
