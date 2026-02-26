"""Unit tests for quiltification.py — SAM + ControlNet image-to-quilt pipeline."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

import backend.services.quiltification as quilt_mod
from backend.services.quiltification import (
    _load_sam,
    _load_controlnet,
    _build_canny_image,
    quiltify_image,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_globals():
    quilt_mod._sam_predictor = None
    quilt_mod._controlnet_pipeline = None


def _make_test_image_bytes(width=64, height=64) -> bytes:
    """Create a simple test image as JPEG bytes."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_fake_pipeline_result():
    """Create a mock ControlNet pipeline result."""
    mock_result = MagicMock()
    mock_result.images = [Image.new("RGB", (1024, 1024), color=(0, 128, 255))]
    return mock_result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_sam
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSam:
    def setup_method(self):
        _reset_globals()

    def test_skips_if_already_loaded(self):
        quilt_mod._sam_predictor = MagicMock()
        _load_sam()
        # Should still be the same mock, not replaced
        assert quilt_mod._sam_predictor is not None

    def test_handles_missing_checkpoint(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_sam = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "segment_anything": mock_sam,
        }):
            with patch("os.path.exists", return_value=False):
                _load_sam()

        assert quilt_mod._sam_predictor is None

    def test_handles_missing_segment_anything(self):
        with patch.dict("sys.modules", {"segment_anything": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                _load_sam()
        assert quilt_mod._sam_predictor is None

    def teardown_method(self):
        _reset_globals()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_controlnet
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadControlnet:
    def setup_method(self):
        _reset_globals()

    def test_skips_if_already_loaded(self):
        quilt_mod._controlnet_pipeline = MagicMock()
        _load_controlnet()
        assert quilt_mod._controlnet_pipeline is not None

    def test_handles_load_failure(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_diffusers = MagicMock()
        mock_diffusers.FluxControlNetModel.from_pretrained.side_effect = RuntimeError("No GPU")

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": mock_diffusers,
        }):
            _load_controlnet()

        assert quilt_mod._controlnet_pipeline is None

    def teardown_method(self):
        _reset_globals()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _build_canny_image
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildCannyImage:
    def setup_method(self):
        _reset_globals()

    def test_returns_pil_image_no_sam_no_cv2(self):
        """Pure numpy gradient fallback path."""
        image_bytes = _make_test_image_bytes()
        with patch.dict("sys.modules", {"cv2": None}):
            result = _build_canny_image(image_bytes)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)

    def test_returns_pil_image_with_cv2(self):
        """OpenCV Canny path (no SAM)."""
        image_bytes = _make_test_image_bytes()

        mock_cv2 = MagicMock()
        mock_cv2.cvtColor.return_value = np.zeros((1024, 1024), dtype=np.uint8)
        mock_cv2.Canny.return_value = np.zeros((1024, 1024), dtype=np.uint8)
        mock_cv2.COLOR_RGB2GRAY = 6

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = _build_canny_image(image_bytes)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)

    def test_returns_pil_image_with_sam(self):
        """SAM boundary path with numpy gradient fallback (no cv2)."""
        image_bytes = _make_test_image_bytes()

        mock_mask = {
            "segmentation": np.ones((1024, 1024), dtype=bool),
        }
        mock_predictor = MagicMock()
        mock_predictor.generate.return_value = [mock_mask]
        quilt_mod._sam_predictor = mock_predictor

        with patch.dict("sys.modules", {"cv2": None}):
            result = _build_canny_image(image_bytes)

        assert result is not None
        assert isinstance(result, Image.Image)
        assert result.size == (1024, 1024)
        mock_predictor.generate.assert_called_once()

    def test_sam_with_cv2_contours(self):
        """SAM boundary path with cv2 contour detection."""
        image_bytes = _make_test_image_bytes()

        mock_mask = {
            "segmentation": np.ones((1024, 1024), dtype=bool),
        }
        mock_predictor = MagicMock()
        mock_predictor.generate.return_value = [mock_mask]
        quilt_mod._sam_predictor = mock_predictor

        mock_cv2 = MagicMock()
        mock_cv2.findContours.return_value = ([], None)
        mock_cv2.RETR_EXTERNAL = 0
        mock_cv2.CHAIN_APPROX_SIMPLE = 1

        with patch.dict("sys.modules", {"cv2": mock_cv2}):
            result = _build_canny_image(image_bytes)

        assert result is not None
        assert isinstance(result, Image.Image)
        mock_cv2.findContours.assert_called()

    def test_resizes_to_1024(self):
        """Input image is resized to 1024x1024 regardless of input size."""
        image_bytes = _make_test_image_bytes(width=200, height=300)

        with patch.dict("sys.modules", {"cv2": None}):
            result = _build_canny_image(image_bytes)

        assert result.size == (1024, 1024)

    def test_returns_none_if_numpy_missing(self):
        """If numpy or PIL are missing, returns None."""
        image_bytes = _make_test_image_bytes()

        with patch.dict("sys.modules", {"numpy": None}):
            with patch("builtins.__import__", side_effect=ImportError("No numpy")):
                result = _build_canny_image(image_bytes)

        assert result is None

    def teardown_method(self):
        _reset_globals()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: quiltify_image
# ─────────────────────────────────────────────────────────────────────────────

class TestQuiltifyImage:
    def setup_method(self):
        _reset_globals()

    def test_returns_none_when_no_pipeline(self):
        """Returns None if ControlNet isn't available."""
        image_bytes = _make_test_image_bytes()

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=Image.new("RGB", (1024, 1024))):
                    result = quiltify_image(image_bytes)

        assert result is None

    def test_returns_none_when_canny_fails(self):
        """Returns None if edge detection fails."""
        image_bytes = _make_test_image_bytes()

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=None):
                    result = quiltify_image(image_bytes)

        assert result is None

    def test_returns_jpeg_bytes(self):
        """Full pipeline returns JPEG bytes when everything is available."""
        image_bytes = _make_test_image_bytes()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_fake_pipeline_result()
        quilt_mod._controlnet_pipeline = mock_pipe

        canny_img = Image.new("RGB", (1024, 1024))

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=canny_img):
                    result = quiltify_image(image_bytes)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG"

    def test_passes_prompt_with_quilt_style(self):
        """Prompt is augmented with pictorial modern quilt style directives."""
        image_bytes = _make_test_image_bytes()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_fake_pipeline_result()
        quilt_mod._controlnet_pipeline = mock_pipe

        canny_img = Image.new("RGB", (1024, 1024))

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=canny_img):
                    quiltify_image(image_bytes, prompt="a golden retriever")

        call_kwargs = mock_pipe.call_args[1]
        assert "golden retriever" in call_kwargs["prompt"]
        assert "pictorial modern quilt" in call_kwargs["prompt"]

    def test_passes_controlnet_params(self):
        """ControlNet conditioning scale and inference params are passed through."""
        image_bytes = _make_test_image_bytes()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_fake_pipeline_result()
        quilt_mod._controlnet_pipeline = mock_pipe

        canny_img = Image.new("RGB", (1024, 1024))

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=canny_img):
                    quiltify_image(
                        image_bytes,
                        controlnet_conditioning_scale=0.8,
                        num_inference_steps=15,
                        guidance_scale=7.0,
                    )

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["controlnet_conditioning_scale"] == 0.8
        assert call_kwargs["num_inference_steps"] == 15
        assert call_kwargs["guidance_scale"] == 7.0

    def test_uses_canny_image_dimensions(self):
        """Pipeline is called with the same width/height as the canny image."""
        image_bytes = _make_test_image_bytes()
        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_fake_pipeline_result()
        quilt_mod._controlnet_pipeline = mock_pipe

        canny_img = Image.new("RGB", (1024, 1024))

        with patch.object(quilt_mod, "_load_sam"):
            with patch.object(quilt_mod, "_load_controlnet"):
                with patch.object(quilt_mod, "_build_canny_image", return_value=canny_img):
                    quiltify_image(image_bytes)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["width"] == 1024
        assert call_kwargs["height"] == 1024

    def teardown_method(self):
        _reset_globals()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
