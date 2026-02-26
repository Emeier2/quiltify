"""
Quiltification Pipeline — turns an input photo into a pictorial modern quilt image.

Pipeline:
  1. SAM (sam-vit-base) segments the input image into regions
  2. Canny edge map is derived from SAM segment boundaries
  3. ControlNet img2img (FLUX.1-dev + Canny) re-renders the image as a quilt
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_sam_predictor = None
_controlnet_pipeline = None


def _load_sam() -> None:
    global _sam_predictor
    if _sam_predictor is not None:
        return
    try:
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        checkpoint_path = os.environ.get(
            "SAM_CHECKPOINT",
            os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth")
        )
        if not os.path.exists(checkpoint_path):
            logger.warning(f"SAM checkpoint not found at {checkpoint_path}. "
                           "Download with: "
                           "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            return

        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        _sam_predictor = SamAutomaticMaskGenerator(sam, points_per_side=16)
        logger.info(f"Loaded SAM vit_b on {device}")
    except ImportError as e:
        logger.warning(f"segment-anything not available: {e}")


def _load_controlnet() -> None:
    global _controlnet_pipeline
    if _controlnet_pipeline is not None:
        return
    try:
        import torch
        from diffusers import FluxControlNetPipeline, FluxControlNetModel

        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
            torch_dtype=torch.float16,
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        _controlnet_pipeline = pipe
        logger.info("Loaded FLUX ControlNet Canny")
    except Exception as e:
        logger.warning(f"ControlNet pipeline load failed: {e}")


def quiltify_image(
    image_bytes: bytes,
    prompt: str = "a modern geometric quilt",
    controlnet_conditioning_scale: float = 0.6,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
) -> Optional[bytes]:
    """
    Transform an input image into a pictorial modern quilt.
    Returns JPEG bytes of the quilt version, or None if unavailable.
    """
    _load_sam()
    _load_controlnet()

    # Build Canny edge image (from SAM boundaries if available, else direct Canny)
    canny_image = _build_canny_image(image_bytes)
    if canny_image is None:
        return None

    if _controlnet_pipeline is None:
        logger.info("ControlNet pipeline unavailable")
        return None

    from PIL import Image

    quilt_prompt = (
        f"{prompt}, pictorial modern quilt, solid fabric geometric squares "
        "and rectangles, bold solid colors, pictorial patchwork, clean grid lines, "
        "quilt top flat view, no gradients no textures no shadows"
    )

    result = _controlnet_pipeline(
        prompt=quilt_prompt,
        control_image=canny_image,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=canny_image.width,
        height=canny_image.height,
    )
    pil_image = result.images[0]

    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _build_canny_image(image_bytes: bytes):
    """
    Build a Canny edge-detection image.
    If SAM is available, derive edges from segment boundaries;
    otherwise run OpenCV Canny directly.
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return None

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Resize to 1024×1024 for ControlNet
    img_pil = img_pil.resize((1024, 1024), Image.LANCZOS)
    img_arr = np.array(img_pil)

    if _sam_predictor is not None:
        masks = _sam_predictor.generate(img_arr)
        # Build edge map from mask boundaries
        edge_map = np.zeros(img_arr.shape[:2], dtype=np.uint8)
        for mask_info in masks:
            mask = mask_info["segmentation"].astype(np.uint8) * 255
            try:
                import cv2
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(edge_map, contours, -1, 255, 1)
            except ImportError:
                # Fallback: use numpy gradient as edge
                gy = np.abs(np.diff(mask.astype(int), axis=0))
                gx = np.abs(np.diff(mask.astype(int), axis=1))
                edge_map[:gy.shape[0], :gy.shape[1]] |= (gy[:, :edge_map.shape[1]] > 0).astype(np.uint8) * 255
                edge_map[:gx.shape[0], :gx.shape[1]] |= (gx[:edge_map.shape[0], :] > 0).astype(np.uint8) * 255
        canny_rgb = np.stack([edge_map] * 3, axis=-1)
    else:
        # Direct Canny on the image
        try:
            import cv2
            gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            canny_rgb = np.stack([edges] * 3, axis=-1)
        except ImportError:
            # Pure numpy gradient
            gray = np.mean(img_arr, axis=2).astype(np.uint8)
            gy = np.abs(np.diff(gray.astype(int), axis=0, append=0)).astype(np.uint8)
            gx = np.abs(np.diff(gray.astype(int), axis=1, append=0)).astype(np.uint8)
            edges = np.clip(gy + gx, 0, 255).astype(np.uint8)
            canny_rgb = np.stack([edges] * 3, axis=-1)

    return Image.fromarray(canny_rgb.astype(np.uint8))
