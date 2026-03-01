"""
vtracer Service — vectorize raster images into clean SVG paths.

Uses vtracer (Rust-based image vectorizer via PyO3) to convert FLUX-generated
raster images into clean vector paths before grid extraction.  This removes
raster noise, anti-aliasing artifacts, and gradient bleed, producing sharper
color boundaries that improve corner detection and block merging.

Install: pip install vtracer
"""
from __future__ import annotations

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import vtracer
    _HAS_VTRACER = True
except ImportError:
    _HAS_VTRACER = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def is_available() -> bool:
    """Check if vtracer and PIL are installed."""
    return _HAS_VTRACER and _HAS_PIL


def vectorize_image(
    image_bytes: bytes,
    color_precision: int = 5,
    filter_speckle: int = 8,
    layer_difference: int = 24,
    mode: str = "polygon",
    corner_threshold: int = 60,
    path_precision: int = 4,
) -> Optional[str]:
    """
    Vectorize a raster image into an SVG string.

    Parameters tuned for quilt-style images:
      - color_precision=5: Fewer distinct colors → cleaner solid regions
      - filter_speckle=8: Remove small artifacts below 8px
      - layer_difference=24: Merge similar color layers aggressively
      - mode="polygon": Straight edges (quilts are geometric, not organic)
      - corner_threshold=60: Preserve sharp corners
      - path_precision=4: Enough precision for grid alignment

    Returns SVG string or None if vtracer is unavailable.
    """
    if not _HAS_VTRACER:
        return None

    try:
        svg_str = vtracer.convert_raw_image_to_svg(
            image_bytes,
            img_format="jpg",
            colormode="color",
            hierarchical="stacked",
            mode=mode,
            filter_speckle=filter_speckle,
            color_precision=color_precision,
            layer_difference=layer_difference,
            corner_threshold=corner_threshold,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=path_precision,
        )
        return svg_str
    except Exception as e:
        logger.warning(f"vtracer vectorization failed: {e}")
        return None


def vectorize_and_rasterize(
    image_bytes: bytes,
    width: int,
    height: int,
    **kwargs,
) -> Optional[bytes]:
    """
    Vectorize a raster image then re-rasterize at the target dimensions.

    This round-trip through vector format removes raster noise, anti-aliasing
    artifacts, and gradient bleed while preserving clean color boundaries.
    Returns JPEG bytes at the target resolution, or None if unavailable.
    """
    if not _HAS_VTRACER or not _HAS_PIL:
        return None

    svg_str = vectorize_image(image_bytes, **kwargs)
    if not svg_str:
        return None

    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            output_width=width,
            output_height=height,
        )
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except ImportError:
        logger.info("cairosvg not available, skipping SVG re-rasterization")
        return None
    except Exception as e:
        logger.warning(f"SVG re-rasterization failed: {e}")
        return None


def clean_for_extraction(
    image_bytes: bytes,
    grid_width: int,
    grid_height: int,
    cell_px: int = 24,
) -> Optional[bytes]:
    """
    Vectorize then re-rasterize at exact grid extraction resolution.

    This is the primary entry point for the grid extraction pipeline.
    Returns cleaned image bytes at (grid_width*cell_px, grid_height*cell_px),
    or None if vtracer/cairosvg are not available (caller falls back to
    original image).
    """
    target_w = grid_width * cell_px
    target_h = grid_height * cell_px
    return vectorize_and_rasterize(image_bytes, target_w, target_h)
