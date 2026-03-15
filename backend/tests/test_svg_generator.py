"""Unit tests for svg_generator.py — StarVector text-to-SVG generation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock

import pytest

import backend.services.svg_generator as svg_mod
from backend.services.svg_generator import (
    generate_quilt_svg,
    generator_status,
    unload,
    _extract_svg,
    STYLE_SUFFIX,
    STARVECTOR_MODEL_ID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reset_generator():
    """Reset module-level generator globals between tests."""
    svg_mod._model = None
    svg_mod._processor = None
    svg_mod._backend = "none"


VALID_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="50" height="50" fill="#ff0000"/>
  <rect x="50" y="50" width="50" height="50" fill="#0000ff"/>
</svg>"""

INVALID_SVG_NO_SHAPES = '<svg viewBox="0 0 100 100"></svg>'


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generator_status
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorStatus:
    def setup_method(self):
        _reset_generator()

    def test_initial_status(self):
        status = generator_status()
        assert status["loaded"] is False
        assert status["type"] == "none"
        assert status["model_id"] == STARVECTOR_MODEL_ID

    def test_status_after_load(self):
        svg_mod._model = MagicMock()
        svg_mod._processor = MagicMock()
        svg_mod._backend = "cuda"
        status = generator_status()
        assert status["loaded"] is True
        assert status["type"] == "cuda"

    def teardown_method(self):
        _reset_generator()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_model
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadModel:
    def setup_method(self):
        _reset_generator()

    def test_skips_if_already_loaded(self):
        svg_mod._model = MagicMock()
        svg_mod._backend = "cuda"
        svg_mod._load_model()
        assert svg_mod._backend == "cuda"

    def test_loads_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock()])
        mock_processor = MagicMock()

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_auto_processor = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.cuda": MagicMock(),
            "transformers": MagicMock(
                AutoModelForCausalLM=mock_auto_model,
                AutoProcessor=mock_auto_processor,
            ),
        }):
            svg_mod._load_model()

        assert svg_mod._backend == "cuda"
        assert svg_mod._model is mock_model
        assert svg_mod._processor is mock_processor
        mock_auto_model.from_pretrained.assert_called_once()
        call_args = mock_auto_model.from_pretrained.call_args
        assert call_args[0][0] == STARVECTOR_MODEL_ID

    def test_skips_when_no_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.cuda": MagicMock(),
            "transformers": MagicMock(),
        }):
            svg_mod._load_model()

        assert svg_mod._backend == "none"
        assert svg_mod._model is None

    def test_handles_import_error(self):
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            svg_mod._load_model()

        assert svg_mod._backend == "none"

    def test_handles_model_load_error(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.side_effect = RuntimeError("OOM")

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "torch.cuda": MagicMock(),
            "transformers": MagicMock(
                AutoModelForCausalLM=mock_auto_model,
                AutoProcessor=MagicMock(),
            ),
        }):
            svg_mod._load_model()

        assert svg_mod._backend == "none"

    def teardown_method(self):
        _reset_generator()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_quilt_svg
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateQuiltSvg:
    def setup_method(self):
        _reset_generator()

    def test_returns_none_when_no_backend(self):
        with patch.object(svg_mod, "_load_model"):
            result = generate_quilt_svg("a sunset quilt")
        assert result is None

    def test_returns_svg_via_text2svg(self):
        mock_model = MagicMock()
        mock_model.generate_text2svg.return_value = VALID_SVG

        svg_mod._model = mock_model
        svg_mod._processor = MagicMock()
        svg_mod._backend = "cuda"

        with patch.object(svg_mod, "_load_model"):
            result = generate_quilt_svg("a sunset quilt")

        assert result is not None
        assert "<svg" in result
        assert "<rect" in result

    def test_strips_non_svg_content(self):
        raw = f"Here is the SVG:\n{VALID_SVG}\nDone!"
        result = _extract_svg(raw)
        assert result is not None
        assert result.startswith("<svg")
        assert result.endswith("</svg>")
        assert "Here is" not in result
        assert "Done!" not in result

    def test_returns_none_on_no_shapes(self):
        result = _extract_svg(INVALID_SVG_NO_SHAPES)
        assert result is None

    def test_returns_none_on_empty_string(self):
        result = _extract_svg("")
        assert result is None

    def test_returns_none_on_no_svg_tags(self):
        result = _extract_svg("just some random text without svg")
        assert result is None

    def test_appends_style_suffix(self):
        mock_model = MagicMock()
        mock_model.generate_text2svg.return_value = VALID_SVG

        svg_mod._model = mock_model
        svg_mod._processor = MagicMock()
        svg_mod._backend = "cuda"

        with patch.object(svg_mod, "_load_model"):
            generate_quilt_svg("a sunset quilt")

        call_args = mock_model.generate_text2svg.call_args
        prompt_used = call_args[0][0]
        assert "a sunset quilt" in prompt_used
        assert "geometric quilt" in prompt_used

    def teardown_method(self):
        _reset_generator()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: unload
# ─────────────────────────────────────────────────────────────────────────────

class TestUnload:
    def setup_method(self):
        _reset_generator()

    def test_unload_resets_state(self):
        svg_mod._model = MagicMock()
        svg_mod._processor = MagicMock()
        svg_mod._backend = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            unload()

        assert svg_mod._model is None
        assert svg_mod._processor is None
        assert svg_mod._backend == "none"
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_unload_safe_when_not_loaded(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            unload()  # Should not raise

        assert svg_mod._backend == "none"

    def teardown_method(self):
        _reset_generator()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Style suffix
# ─────────────────────────────────────────────────────────────────────────────

class TestStyleSuffix:
    def test_contains_quilt_keywords(self):
        assert "quilt" in STYLE_SUFFIX
        assert "geometric" in STYLE_SUFFIX
        assert "solid" in STYLE_SUFFIX

    def test_starts_with_comma(self):
        assert STYLE_SUFFIX.startswith(",")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
