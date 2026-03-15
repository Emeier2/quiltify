"""Unit tests for flux_pipeline.py — multi-backend FLUX text-to-image generation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import base64
import io
import json
from unittest.mock import patch, MagicMock, PropertyMock

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
    flux_mod._backend = "none"
    flux_mod._forge_url = None


def _make_fake_image() -> Image.Image:
    """Create a small 4x4 RGB image for mock pipeline output."""
    return Image.new("RGB", (4, 4), color=(128, 64, 32))


def _make_fake_png_base64() -> str:
    """Create a base64-encoded PNG for mock forge API response."""
    img = _make_fake_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


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

    def test_status_after_cuda_load(self):
        flux_mod._pipeline = MagicMock()
        flux_mod._backend = "cuda"
        status = pipeline_status()
        assert status["loaded"] is True
        assert status["type"] == "cuda"

    def test_status_after_forge_load(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"
        status = pipeline_status()
        assert status["loaded"] is True
        assert status["type"] == "forge-api"

    def test_status_after_directml_load(self):
        flux_mod._pipeline = MagicMock()
        flux_mod._backend = "directml"
        status = pipeline_status()
        assert status["loaded"] is True
        assert status["type"] == "directml"

    def test_status_after_schnell_cpu_load(self):
        flux_mod._pipeline = MagicMock()
        flux_mod._backend = "schnell-cpu"
        status = pipeline_status()
        assert status["loaded"] is True
        assert status["type"] == "schnell-cpu"

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
        flux_mod._backend = "cuda"
        flux_mod._load_pipeline()
        assert flux_mod._backend == "cuda"

    def test_skips_if_forge_already_loaded(self):
        """forge-api has no _pipeline object but _backend is set — should skip."""
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"
        flux_mod._load_pipeline()
        assert flux_mod._backend == "forge-api"

    def test_loads_cuda_when_available(self):
        mock_pipe = _make_mock_pipeline()
        mock_torch = MagicMock()
        mock_torch.float16 = "float16"
        mock_torch.cuda.is_available.return_value = True

        mock_flux_cls = MagicMock()
        mock_flux_cls.from_pretrained.return_value = mock_pipe

        mock_bnb_config = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.cuda": MagicMock(),
            "diffusers": MagicMock(FluxPipeline=mock_flux_cls, BitsAndBytesConfig=mock_bnb_config),
        }):
            flux_mod._load_pipeline()

        assert flux_mod._pipeline is mock_pipe
        assert flux_mod._backend == "cuda"
        mock_flux_cls.from_pretrained.assert_called_once()
        call_args = mock_flux_cls.from_pretrained.call_args
        assert call_args[0][0] == FLUX_MODEL_ID

    def test_falls_back_to_none_when_all_fail(self):
        with patch.object(flux_mod, "_try_cuda", return_value=False), \
             patch.object(flux_mod, "_try_forge_api", return_value=False), \
             patch.object(flux_mod, "_try_directml", return_value=False), \
             patch.object(flux_mod, "_try_schnell_cpu", return_value=False):
            flux_mod._load_pipeline()

        assert flux_mod._backend == "none"

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Forge API backend
# ─────────────────────────────────────────────────────────────────────────────

class TestForgeApiBackend:
    def setup_method(self):
        _reset_pipeline()

    def test_try_forge_detects_running_server(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._try_forge_api()

        assert result is True
        assert flux_mod._backend == "forge-api"
        assert flux_mod._forge_url is not None
        mock_httpx.get.assert_called_once()

    def test_try_forge_returns_false_when_unreachable(self):
        mock_httpx = MagicMock()
        mock_httpx.get.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._try_forge_api()

        assert result is False
        assert flux_mod._backend == "none"

    def test_try_forge_returns_false_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        mock_httpx = MagicMock()
        mock_httpx.get.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._try_forge_api()

        assert result is False

    def test_generate_via_forge_sends_correct_payload(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"

        b64_img = _make_fake_png_base64()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": [b64_img]}
        mock_resp.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._generate_via_forge(
                "test prompt", 512, 768, 20, 3.5, 42
            )

        assert result is not None
        # Verify POST was called with correct payload
        call_args = mock_httpx.post.call_args
        payload = call_args[1]["json"]
        assert payload["prompt"] == "test prompt"
        assert payload["width"] == 512
        assert payload["height"] == 768
        assert payload["steps"] == 20
        assert payload["cfg_scale"] == 3.5
        assert payload["seed"] == 42

    def test_generate_via_forge_uses_negative_one_seed_when_none(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"

        b64_img = _make_fake_png_base64()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": [b64_img]}
        mock_resp.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            flux_mod._generate_via_forge("test", 512, 512, 20, 3.5, None)

        payload = mock_httpx.post.call_args[1]["json"]
        assert payload["seed"] == -1

    def test_generate_via_forge_returns_jpeg(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"

        b64_img = _make_fake_png_base64()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": [b64_img]}
        mock_resp.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._generate_via_forge("test", 512, 512, 20, 3.5, None)

        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG"

    def test_generate_via_forge_returns_none_on_error(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"

        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = Exception("Connection lost")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = flux_mod._generate_via_forge("test", 512, 512, 20, 3.5, None)

        assert result is None

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: DirectML backend
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectMLBackend:
    def setup_method(self):
        _reset_pipeline()

    def test_try_directml_loads_with_dml_provider(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]

        mock_pipe = _make_mock_pipeline()
        mock_ort_flux = MagicMock()
        mock_ort_flux.from_pretrained.return_value = mock_pipe

        mock_optimum = MagicMock()

        with patch.dict("sys.modules", {
            "onnxruntime": mock_ort,
            "optimum": mock_optimum,
            "optimum.onnxruntime": MagicMock(ORTFluxPipeline=mock_ort_flux),
        }):
            result = flux_mod._try_directml()

        assert result is True
        assert flux_mod._backend == "directml"
        assert flux_mod._pipeline is mock_pipe
        mock_ort_flux.from_pretrained.assert_called_once()
        call_kwargs = mock_ort_flux.from_pretrained.call_args
        assert call_kwargs[1]["provider"] == "DmlExecutionProvider"

    def test_try_directml_skips_when_no_dml_provider(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            result = flux_mod._try_directml()

        assert result is False
        assert flux_mod._backend == "none"

    def test_try_directml_returns_false_when_not_installed(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                result = flux_mod._try_directml()

        assert result is False
        assert flux_mod._backend == "none"

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Backend priority
# ─────────────────────────────────────────────────────────────────────────────

class TestBackendPriority:
    def setup_method(self):
        _reset_pipeline()

    def test_cuda_tried_first(self):
        """If CUDA succeeds, later backends are not tried."""
        call_order = []

        def mock_cuda():
            call_order.append("cuda")
            flux_mod._backend = "cuda"
            return True

        def mock_forge():
            call_order.append("forge")
            return True

        with patch.object(flux_mod, "_try_cuda", side_effect=mock_cuda), \
             patch.object(flux_mod, "_try_forge_api", side_effect=mock_forge), \
             patch.object(flux_mod, "_try_directml", return_value=False), \
             patch.object(flux_mod, "_try_schnell_cpu", return_value=False):
            flux_mod._load_pipeline()

        assert call_order == ["cuda"]
        assert flux_mod._backend == "cuda"

    def test_forge_tried_after_cuda_fails(self):
        call_order = []

        def mock_cuda():
            call_order.append("cuda")
            return False

        def mock_forge():
            call_order.append("forge")
            flux_mod._backend = "forge-api"
            return True

        with patch.object(flux_mod, "_try_cuda", side_effect=mock_cuda), \
             patch.object(flux_mod, "_try_forge_api", side_effect=mock_forge), \
             patch.object(flux_mod, "_try_directml", return_value=False), \
             patch.object(flux_mod, "_try_schnell_cpu", return_value=False):
            flux_mod._load_pipeline()

        assert call_order == ["cuda", "forge"]
        assert flux_mod._backend == "forge-api"

    def test_directml_tried_after_forge_fails(self):
        call_order = []

        def mock_cuda():
            call_order.append("cuda")
            return False

        def mock_forge():
            call_order.append("forge")
            return False

        def mock_dml():
            call_order.append("directml")
            flux_mod._backend = "directml"
            return True

        with patch.object(flux_mod, "_try_cuda", side_effect=mock_cuda), \
             patch.object(flux_mod, "_try_forge_api", side_effect=mock_forge), \
             patch.object(flux_mod, "_try_directml", side_effect=mock_dml), \
             patch.object(flux_mod, "_try_schnell_cpu", return_value=False):
            flux_mod._load_pipeline()

        assert call_order == ["cuda", "forge", "directml"]
        assert flux_mod._backend == "directml"

    def test_schnell_cpu_is_last_resort(self):
        call_order = []

        def mock_cuda():
            call_order.append("cuda")
            return False

        def mock_forge():
            call_order.append("forge")
            return False

        def mock_dml():
            call_order.append("directml")
            return False

        def mock_schnell():
            call_order.append("schnell")
            flux_mod._backend = "schnell-cpu"
            return True

        with patch.object(flux_mod, "_try_cuda", side_effect=mock_cuda), \
             patch.object(flux_mod, "_try_forge_api", side_effect=mock_forge), \
             patch.object(flux_mod, "_try_directml", side_effect=mock_dml), \
             patch.object(flux_mod, "_try_schnell_cpu", side_effect=mock_schnell):
            flux_mod._load_pipeline()

        assert call_order == ["cuda", "forge", "directml", "schnell"]
        assert flux_mod._backend == "schnell-cpu"

    def test_all_fail_results_in_none(self):
        with patch.object(flux_mod, "_try_cuda", return_value=False), \
             patch.object(flux_mod, "_try_forge_api", return_value=False), \
             patch.object(flux_mod, "_try_directml", return_value=False), \
             patch.object(flux_mod, "_try_schnell_cpu", return_value=False):
            flux_mod._load_pipeline()

        assert flux_mod._backend == "none"
        assert flux_mod._pipeline is None

    def teardown_method(self):
        _reset_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_quilt_image
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateQuiltImage:
    def setup_method(self):
        _reset_pipeline()

    def test_returns_none_when_no_backend(self):
        with patch.object(flux_mod, "_load_pipeline"):
            result = generate_quilt_image("a cat quilt")
        assert result is None

    def test_returns_jpeg_bytes_via_local(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        flux_mod._backend = "cuda"
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                result = generate_quilt_image("a cat quilt")

        assert result is not None
        assert isinstance(result, bytes)
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG"

    def test_dispatches_to_forge_api(self):
        flux_mod._backend = "forge-api"
        flux_mod._forge_url = "http://localhost:7860"

        b64_img = _make_fake_png_base64()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"images": [b64_img]}
        mock_resp.raise_for_status = MagicMock()

        mock_httpx = MagicMock()
        mock_httpx.post.return_value = mock_resp

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"httpx": mock_httpx}):
                result = generate_quilt_image("a cat quilt")

        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert img.format == "JPEG"
        # Verify prompt had style suffix appended
        payload = mock_httpx.post.call_args[1]["json"]
        assert payload["prompt"] == "a cat quilt" + STYLE_SUFFIX

    def test_appends_style_suffix_local(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        flux_mod._backend = "cuda"
        mock_torch = MagicMock()

        with patch.object(flux_mod, "_load_pipeline"):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                generate_quilt_image("a cat quilt")

        call_kwargs = mock_pipe.call_args[1]
        assert call_kwargs["prompt"] == "a cat quilt" + STYLE_SUFFIX

    def test_passes_dimensions(self):
        mock_pipe = _make_mock_pipeline()
        flux_mod._pipeline = mock_pipe
        flux_mod._backend = "cuda"
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
        flux_mod._backend = "cuda"
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
        flux_mod._backend = "cuda"

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
        flux_mod._backend = "cuda"
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
