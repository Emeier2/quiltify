"""
FLUX Pipeline — text-to-image generation using FLUX.1-dev.

Supports multiple backends, tried in priority order:
  1. CUDA + bitsandbytes — NVIDIA GPUs, Q4 quantized FLUX.1-dev
  2. webui-forge API     — external Stable Diffusion WebUI Forge process (AMD via DirectML)
  3. ORTFluxPipeline     — ONNX Runtime + DirectML (AMD GPUs, direct Python)
  4. FLUX.1-schnell CPU  — slow but works everywhere
  5. "none"              — graceful degradation

Requires (for full CUDA path):
  pip install diffusers transformers accelerate bitsandbytes torch
  A HuggingFace token with access to black-forest-labs/FLUX.1-dev
"""
from __future__ import annotations

import base64
import io
import os
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Model IDs
FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_MODEL_ID_SCHNELL = "black-forest-labs/FLUX.1-schnell"  # fallback, no auth needed

# Style suffix appended to all prompts
STYLE_SUFFIX = (
    ", pictorial modern quilt pattern, solid fabric geometric squares and rectangles, "
    "bold solid colors, pictorial patchwork design, clean grid lines, quilt top view, "
    "no gradients no shading no textures"
)

# Env-configurable settings
FORGE_API_URL = os.environ.get("FORGE_API_URL", "http://localhost:7860")
FLUX_ONNX_MODEL = os.environ.get("FLUX_ONNX_MODEL", "IlyasMoutawwakil/flux-onnx-optimum")
FLUX_ONNX_LOCAL_DIR = os.environ.get(
    "FLUX_ONNX_LOCAL_DIR",
    str(Path(__file__).resolve().parents[1] / ".hf" / "flux-onnx-optimum"),
)
GPU_ONLY = os.environ.get("GPU_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}

# Module-level state
_backend: str = "none"  # "cuda", "forge-api", "directml", "schnell-cpu", "none"
_pipeline = None         # Local pipeline object (cuda/directml/cpu)
_forge_url: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Backend loaders
# ─────────────────────────────────────────────────────────────────────────────

def _try_cuda() -> bool:
    """Try CUDA + BitsAndBytes Q4 quantized FLUX.1-dev."""
    global _pipeline, _backend
    try:
        import torch
        if not torch.cuda.is_available():
            logger.info("CUDA not available, skipping CUDA backend")
            return False

        from diffusers import FluxPipeline, BitsAndBytesConfig

        hf_token = os.environ.get("HF_TOKEN")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL_ID,
            quantization_config=nf4_config,
            torch_dtype=torch.float16,
            token=hf_token,
        )
        if GPU_ONLY:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        _pipeline = pipe
        _backend = "cuda"
        logger.info("Loaded FLUX.1-dev (Q4 CUDA)")
        return True
    except ImportError as e:
        logger.info(f"CUDA backend unavailable (import): {e}")
        return False
    except Exception as e:
        logger.warning(f"CUDA backend failed: {e}")
        return False


def _try_forge_api() -> bool:
    """Check if a webui-forge instance is running and reachable."""
    global _backend, _forge_url
    try:
        import httpx

        url = FORGE_API_URL.rstrip("/")
        resp = httpx.get(f"{url}/sdapi/v1/sd-models", timeout=5.0)
        if resp.status_code == 200:
            _forge_url = url
            _backend = "forge-api"
            logger.info(f"Using webui-forge API at {url}")
            return True
        logger.info(f"Forge API returned status {resp.status_code}")
        return False
    except ImportError:
        logger.info("httpx not installed, skipping forge-api backend")
        return False
    except Exception as e:
        logger.info(f"Forge API not reachable: {e}")
        return False


def _download_onnx_repo() -> Path:
    """Download ONNX model repo to a real local folder and return path."""
    from huggingface_hub import snapshot_download

    local_model_dir = Path(FLUX_ONNX_LOCAL_DIR)
    local_model_dir.mkdir(parents=True, exist_ok=True)

    # On Windows without symlink privilege, force real files in a local dir.
    snapshot_download(
        repo_id=FLUX_ONNX_MODEL,
        local_dir=str(local_model_dir),
        local_dir_use_symlinks=False,
    )
    return local_model_dir


def _try_directml() -> bool:
    """Try ONNX Runtime with DirectML execution provider."""
    global _pipeline, _backend
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" not in providers:
            logger.info("DmlExecutionProvider not available, skipping DirectML backend")
            return False

        from optimum.onnxruntime import ORTFluxPipeline

        local_model_dir = _download_onnx_repo()

        pipe = ORTFluxPipeline.from_pretrained(
            local_model_dir.as_posix(),
            provider="DmlExecutionProvider",
        )
        _pipeline = pipe
        _backend = "directml"
        logger.info(f"Loaded ONNX FLUX via DirectML ({local_model_dir})")
        return True
    except ImportError as e:
        logger.info(f"DirectML backend unavailable (import): {e}")
        return False
    except Exception as e:
        logger.warning(f"DirectML backend failed: {e}")
        return False


def _try_onnx_cpu() -> bool:
    """Fallback: ONNX Runtime on CPU using the same FLUX ONNX export."""
    global _pipeline, _backend
    try:
        from optimum.onnxruntime import ORTFluxPipeline

        local_model_dir = _download_onnx_repo()
        pipe = ORTFluxPipeline.from_pretrained(
            local_model_dir.as_posix(),
            provider="CPUExecutionProvider",
        )
        _pipeline = pipe
        _backend = "onnx-cpu"
        logger.info(f"Loaded ONNX FLUX via CPU ({local_model_dir})")
        return True
    except ImportError as e:
        logger.info(f"ONNX CPU backend unavailable (import): {e}")
        return False
    except Exception as e:
        logger.warning(f"ONNX CPU backend failed: {e}")
        return False


def _try_schnell_cpu() -> bool:
    """Fallback: FLUX.1-schnell on CPU. Slow but works everywhere."""
    global _pipeline, _backend
    try:
        import torch
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL_ID_SCHNELL,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        _pipeline = pipe
        _backend = "schnell-cpu"
        logger.info("Loaded FLUX.1-schnell (CPU fallback)")
        return True
    except ImportError as e:
        logger.info(f"Schnell CPU backend unavailable (import): {e}")
        return False
    except Exception as e:
        logger.warning(f"Schnell CPU backend failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline() -> None:
    global _pipeline, _backend, _forge_url

    if _backend != "none":
        return

    loaders = (_try_cuda, _try_forge_api, _try_directml) if GPU_ONLY else (
        _try_cuda, _try_forge_api, _try_directml, _try_onnx_cpu, _try_schnell_cpu
    )
    for loader in loaders:
        if loader():
            return

    _backend = "none"
    logger.warning("No image generation backend available")


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_via_forge(
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> Optional[bytes]:
    """Generate an image via the webui-forge REST API."""
    import httpx

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": num_inference_steps,
        "cfg_scale": guidance_scale,
        "seed": seed if seed is not None else -1,
    }
    try:
        resp = httpx.post(
            f"{_forge_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=1800.0,
        )
        resp.raise_for_status()
        data = resp.json()
        b64_png = data["images"][0]
        png_bytes = base64.b64decode(b64_png)

        # Convert PNG to JPEG for consistency with local backends
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Forge API generation failed: {e}")
        return None


def _generate_local(
    prompt: str,
    width: int,
    height: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> Optional[bytes]:
    """Generate an image via a local pipeline (CUDA, DirectML, or CPU)."""
    import torch

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    kwargs: dict = {
        "prompt": prompt,
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

    if _backend == "none":
        if GPU_ONLY:
            logger.error("GPU_ONLY is enabled and no GPU backend is available")
        else:
            logger.info("No image generation backend available, returning None")
        return None

    full_prompt = prompt + STYLE_SUFFIX

    if _backend == "forge-api":
        return _generate_via_forge(full_prompt, width, height, num_inference_steps, guidance_scale, seed)
    else:
        return _generate_local(full_prompt, width, height, num_inference_steps, guidance_scale, seed)


def pipeline_status() -> dict:
    return {
        "loaded": _backend != "none",
        "type": _backend,
    }
