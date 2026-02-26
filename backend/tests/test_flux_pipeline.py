"""Unit tests for flux_pipeline.py — FLUX text-to-image generation with mocked GPU deps."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import io
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

import backend.services.flux_pipeline as flux_mod
from backend.services.flux_pipeline import (
    generate_quilt_image,
    pipeline_status,
    STYLE_SUFFIX,
    FLUX_MODEL_ID,
    FLUX_MODEL_ID_SCHNELL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_pipeline():
    """Reset the module-level pipeline globals between tests."""
    flux_mod._pipeline = None
    flux_mod._pipeline_type = "none"


def _make_fake_image() -> Image.Image:
    """Create a small 4x4 RGB image for mock pipeline output."""
    return Image.new("RGB", (4, 4), color=(128, 64, 32))


def _make_mock_pipeline():
    """Create a mock pipeline that returns a fake image."""
    mock_pipe = MagicMock()
    result = MagicMock()
    result.images = [_make_fake_image()]
    mock_pipe.return_value = result
    return mock_pipe


# ─────────────────────────────────────────────────────────────────────────────
# Tests: pipeline_status
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineStatus:
    def setup_method(self):
        _reset_pipeline()

    def test_initial_status(self):
        status = pipeline_status()
        assert status["loaded"] is False
        assert status["type"] == "none"

    def test_status_after_load(self):
        flux_mod._pipeline = MagicMock()
        flux_mod._pipeline_type = "flux-dev-q4"
        status = pipeline_status()
        assert status["loaded"] is True
        assert status["type"] == "flux-dev-q4"

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Style suffix
# ─────────────────────────────────────────────────────────────────────────────

class TestStyleSuffix:
    def test_suffix_contains_quilt_keywords(self):
        assert "quilt" in STYLE_SUFFIX
        assert "solid" in STYLE_SUFFIX
        assert "pictorial" in STYLE_SUFFIX

    def test_suffix_starts_with_comma(self):
        assert STYLE_SUFFIX.startswith(",")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPipeline:
    def setup_method(self):
        _reset_pipeline()

    def test_skips_if_already_loaded(self):
        flux_mod._pipeline = MagicMock()
        flux_mod._pipeline_type = "flux-dev-q4"
        # _load_pipeline should return early without touching imports
        flux_mod._load_pipeline()
        assert flux_mod._pipeline_type == "flux-dev-q4"

    def test_loads_dev_q4_when_available(self):
        mock_pipe = _make_mock_pipeline()
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"

        mock_flux_cls = MagicMock()
        mock_flux_cls.from_pretrained.return_value = mock_pipe

        mock_bnb_config = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": MagicMock(FluxPipeline=mock_flux_cls, BitsAndBytesConfig=mock_bnb_config),
        }):
            flux_mod._load_pipeline()

        assert flux_mod._pipeline is mock_pipe
        assert flux_mod._pipeline_type == "flux-dev-q4"
        mock_flux_cls.from_pretrained.assert_called_once()
        call_args = mock_flux_cls.from_pretrained.call_args
        assert call_args[0][0] == FLUX_MODEL_ID

    def test_falls_back_to_schnell(self):
        mock_schnell_pipe = _make_mock_pipeline()
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        mock_flux_cls = MagicMock()
        # Dev fails, schnell succeeds
        mock_flux_cls.from_pretrained.side_effect = [
            RuntimeError("No HF token"),
            mock_schnell_pipe,
        ]

        mock_bnb_config = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": MagicMock(FluxPipeline=mock_flux_cls, BitsAndBytesConfig=mock_bnb_config),
        }):
            flux_mod._load_pipeline()

        assert flux_mod._pipeline is mock_schnell_pipe
        assert flux_mod._pipeline_type == "flux-schnell"
        # Called twice: once for dev, once for schnell
        assert mock_flux_cls.from_pretrained.call_count == 2
        second_call = mock_flux_cls.from_pretrained.call_args_list[1]
        assert second_call[0][0] == FLUX_MODEL_ID_SCHNELL

    def test_falls_back_to_none_when_both_fail(self):
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        mock_flux_cls = MagicMock()
        mock_flux_cls.from_pretrained.side_effect = RuntimeError("Nope")

        mock_bnb_config = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "diffusers": MagicMock(FluxPipeline=mock_flux_cls, BitsAndBytesConfig=mock_bnb_config),
        }):
            flux_mod._load_pipeline()

        assert flux_mod._pipeline is None
        assert flux_mod._pipeline_type == "none"

    def test_handles_missing_diffusers(self):
        """If diffusers/torch aren't installed, pipeline stays None."""
        with patch.dict("sys.modules", {"torch": None, "diffusers": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                flux_mod._load_pipeline()

        assert flux_mod._pipeline is None
        assert flux_mod._pipeline_type == "none"

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_quilt_image
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateQuiltImage:
    def setup_method(self):
        _reset_pipeline()

    def test_returns_none_when_no_pipeline(self):
        # Patch _load_pipeline to do nothing (keep pipeline as None)
        with patch.object(flux_mod, "_load_pipeline"):
            result = generate_quilt_image("a cat quilt")
        assert result is None

    def test_returns_jpeg_bytes(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                result = generate_quilt_image("a cat quilt")

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
        # Verify it's valid JPEG
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG"

    def test_appends_hartman_suffix(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("a cat quilt")

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["prompt"] == "a cat quilt" + STYLE_SUFFIX

    def test_passes_dimensions(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("test", width=512, height=768)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 768

    def test_passes_inference_params(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("test", num_inference_steps=10, guidance_scale=5.0)

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["num_inference_steps"] == 10
        assert call_kwargs["guidance_scale"] == 5.0

    def test_seed_creates_generator(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe

        mock_generator = MagicMock()
        mock_torch = MagicMock()
        mock_torch.Generator.return_value.manual_seed.return_value = mock_generator

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("test", seed=42)

        mock_torch.Generator.return_value.manual_seed.assert_called_once_with(42)
        call_kwargs = mock_pipe.call_args[1]
        assert "generator" in call_kwargs

    def test_no_seed_no_generator(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("test", seed=None)

        call_kwargs = mock_pipe.call_args[1]
        assert "generator" not in call_kwargs

    def teardown_method(self):
        _reset_pipeline()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
