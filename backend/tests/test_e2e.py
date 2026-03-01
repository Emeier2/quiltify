"""
End-to-end tests for the Quiltify API.

Tests exercise the full FastAPI routes with external services mocked:
  - FLUX pipeline → returns a synthetic image or None
  - Ollama client → returns canned guide/layout text
  - Quiltification (SAM + ControlNet) → returns synthetic image bytes

Each test hits the actual HTTP endpoint via TestClient, exercises the full
router → service → domain model → renderer pipeline, and validates the
response structure and data integrity.
"""
from __future__ import annotations

import base64
import csv
import io
import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI TestClient with lifespan events mocked out."""
    with patch("backend.main.ollama_client.check_health", new_callable=AsyncMock, return_value=True):
        with patch("backend.main.flux_pipeline.pipeline_status", return_value={"loaded": False, "type": "none"}):
            with TestClient(app) as c:
                yield c


def _make_test_image(width: int = 100, height: int = 100, color: str = "red") -> bytes:
    """Create a small solid-color JPEG for testing."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _make_test_image_b64(width: int = 100, height: int = 100, color: str = "red") -> str:
    return base64.b64encode(_make_test_image(width, height, color)).decode()


def _make_striped_image_b64() -> str:
    """Create a 100x100 image with 3 horizontal color bands for richer extraction."""
    img = Image.new("RGB", (100, 100))
    for y in range(100):
        for x in range(100):
            if y < 33:
                img.putpixel((x, y), (200, 0, 0))       # red
            elif y < 66:
                img.putpixel((x, y), (0, 0, 200))       # blue
            else:
                img.putpixel((x, y), (0, 200, 0))        # green
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def _make_valid_pattern_payload(
    grid_width: int = 10,
    grid_height: int = 10,
    palette_size: int = 2,
) -> dict:
    """Build a valid QuiltPatternSchema dict for export/guide endpoints."""
    fabrics = [
        {"id": "f1", "color_hex": "#cc0000", "name": "Kona Cotton - Red", "total_sqin": 0.0},
        {"id": "f2", "color_hex": "#0000cc", "name": "Kona Cotton - Blue", "total_sqin": 0.0},
    ][:palette_size]

    # Simple two-band pattern covering the full grid
    mid = grid_height // 2
    blocks = [
        {"x": 0, "y": 0, "width": grid_width, "height": mid, "fabric_id": "f1"},
        {"x": 0, "y": mid, "width": grid_width, "height": grid_height - mid, "fabric_id": "f2"},
    ]

    cell_sizes = [{"w": 2.5, "h": 2.5} for _ in range(grid_width * grid_height)]
    return {
        "grid_width": grid_width,
        "grid_height": grid_height,
        "quilt_width_in": grid_width * 2.5,
        "quilt_height_in": grid_height * 2.5,
        "seam_allowance": 0.25,
        "fabrics": fabrics,
        "blocks": blocks,
        "cell_sizes": cell_sizes,
    }


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_ok(self, client):
        with patch("backend.main.ollama_client.check_health", new_callable=AsyncMock, return_value=True):
            resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "ollama" in data
        assert "flux_pipeline" in data

    def test_ollama_down(self, client):
        with patch("backend.main.ollama_client.check_health", new_callable=AsyncMock, return_value=False):
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["ollama"] is False


# ---------------------------------------------------------------------------
# /api/generate — full pipeline: prompt → pattern
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    """Test the text-prompt → quilt pattern pipeline."""

    def test_synthetic_fallback_when_no_flux_no_ollama(self, client):
        """With no FLUX and Ollama failing, should return synthetic stripes."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image", return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("Ollama down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Test guide text"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "a mountain sunset quilt",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
                "quilt_width_in": 25.0,
                "quilt_height_in": 25.0,
            })

        assert resp.status_code == 200
        data = resp.json()

        # Structure checks
        assert "pattern_json" in data
        assert "svg" in data
        assert "cutting_svg" in data
        assert "cutting_chart" in data
        assert "guide" in data
        assert "confidence_score" in data
        assert "validation_errors" in data
        assert "pipeline_status" in data

        # Synthetic fallback → confidence 0.0
        assert data["confidence_score"] == 0.0
        assert data["image_b64"] is None

        # Pattern integrity
        pj = data["pattern_json"]
        assert pj["grid_width"] == 10
        assert pj["grid_height"] == 10
        assert len(pj["fabrics"]) == 3
        assert len(pj["blocks"]) > 0
        assert data["validation_errors"] == []

    def test_with_flux_image(self, client):
        """When FLUX returns an image, grid extractor should process it."""
        test_image = _make_test_image(100, 100, "blue")

        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=test_image), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Guide from Ollama"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": True, "type": "flux-dev-q4"}):

            resp = client.post("/api/generate", json={
                "prompt": "a forest quilt",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 4,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_score"] > 0
        assert data["image_b64"] is not None
        assert data["guide"] == "Guide from Ollama"
        assert data["validation_errors"] == []

        # image_b64 should be valid base64 of the original image
        decoded = base64.b64decode(data["image_b64"])
        assert len(decoded) == len(test_image)

    def test_ollama_layout_fallback(self, client):
        """When FLUX unavailable, Ollama layout should be used if valid."""
        layout_json = {
            "fabrics": [
                {"id": "f1", "color_hex": "#ff0000", "name": "Red"},
                {"id": "f2", "color_hex": "#00ff00", "name": "Green"},
            ],
            "blocks": [
                {"x": 0, "y": 0, "width": 10, "height": 5, "fabric_id": "f1"},
                {"x": 0, "y": 5, "width": 10, "height": 5, "fabric_id": "f2"},
            ],
            "cell_sizes": [{"w": 2.5, "h": 2.5} for _ in range(100)],
        }

        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, return_value=layout_json), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Ollama guide"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "flower garden",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 2,
                "quilt_width_in": 25.0,
                "quilt_height_in": 25.0,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence_score"] == 0.5
        assert len(data["pattern_json"]["fabrics"]) == 2
        assert data["validation_errors"] == []

    def test_ollama_layout_invalid_falls_through_to_synthetic(self, client):
        """Ollama layout that fails validation should fall through to synthetic."""
        bad_layout = {
            "fabrics": [{"id": "f1", "color_hex": "#ff0000", "name": "Red"}],
            "blocks": [
                # Doesn't cover the whole grid — validation will fail
                {"x": 0, "y": 0, "width": 5, "height": 5, "fabric_id": "f1"},
            ],
        }

        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, return_value=bad_layout), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="guide"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        assert resp.status_code == 200
        data = resp.json()
        # Fell through to synthetic stripes
        assert data["confidence_score"] == 0.0

    def test_guide_fallback_on_ollama_error(self, client):
        """When Ollama guide fails, guide should be cutting instructions."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, side_effect=Exception("Ollama timeout")), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={"prompt": "test"})

        assert resp.status_code == 200
        data = resp.json()
        # Guide should be the cutting instructions joined with newlines
        assert "###" in data["guide"]  # format_cutting_sequence uses ### headers

    def test_svg_output_is_valid(self, client):
        """SVG fields should contain valid SVG markup."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="guide"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test",
                "grid_width": 10,
                "grid_height": 10,
            })

        data = resp.json()
        assert "<svg" in data["svg"]
        assert "</svg>" in data["svg"]
        assert "<svg" in data["cutting_svg"]
        assert "</svg>" in data["cutting_svg"]

    def test_cutting_chart_structure(self, client):
        """Cutting chart entries should have all required fields."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="guide"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        chart = resp.json()["cutting_chart"]
        assert len(chart) > 0
        for piece in chart:
            assert "fabric_id" in piece
            assert "fabric_name" in piece
            assert "color_hex" in piece
            assert "cut_width_in" in piece
            assert "cut_height_in" in piece
            assert "quantity" in piece
            assert piece["quantity"] > 0
            assert piece["cut_width_in"] > 0
            assert piece["cut_height_in"] > 0

    def test_request_validation_rejects_bad_params(self, client):
        """Pydantic validation should reject out-of-range parameters."""
        resp = client.post("/api/generate", json={
            "prompt": "test",
            "grid_width": 5,  # min is 10
        })
        assert resp.status_code == 422

        resp = client.post("/api/generate", json={
            "prompt": "test",
            "palette_size": 0,  # min is 2
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/quiltify — full pipeline: image → quilt pattern
# ---------------------------------------------------------------------------

class TestQuiltifyEndpoint:
    """Test the image → quilt pattern pipeline."""

    def test_basic_quiltify_with_quiltification_fallback(self, client):
        """When SAM/ControlNet unavailable, should use original image for extraction."""
        image_b64 = _make_striped_image_b64()

        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   side_effect=Exception("SAM not available")), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Test guide"):

            resp = client.post("/api/quiltify", json={
                "image_base64": image_b64,
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        assert resp.status_code == 200
        data = resp.json()

        # Structure
        assert "pattern_json" in data
        assert "svg" in data
        assert "cutting_svg" in data
        assert "cutting_chart" in data
        assert "guide" in data
        assert "confidence_score" in data
        assert "validation_errors" in data
        assert "original_image_b64" in data
        assert "quilt_image_b64" in data

        # Pattern should be valid
        assert data["validation_errors"] == []
        pj = data["pattern_json"]
        assert pj["grid_width"] == 10
        assert pj["grid_height"] == 10

        # Should have extracted multiple fabrics from the striped image
        assert len(pj["fabrics"]) >= 2

    def test_quiltify_with_successful_quiltification(self, client):
        """When SAM/ControlNet works, quilt_image_b64 should be populated."""
        original_b64 = _make_test_image_b64(100, 100, "red")
        quilt_image = _make_test_image(100, 100, "blue")

        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   return_value=quilt_image), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Guide text"):

            resp = client.post("/api/quiltify", json={
                "image_base64": original_b64,
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 4,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["original_image_b64"] is not None
        assert data["quilt_image_b64"] is not None
        # Quilt image should differ from original
        assert data["original_image_b64"] != data["quilt_image_b64"]

    def test_invalid_base64_returns_400(self, client):
        """Bad base64 data should return 400."""
        resp = client.post("/api/quiltify", json={
            "image_base64": "not-valid-base64!!!",
            "grid_width": 10,
            "grid_height": 10,
        })
        assert resp.status_code == 400

    def test_data_uri_prefix_handled(self, client):
        """image_base64 with data:image/jpeg;base64, prefix should work."""
        raw_b64 = _make_striped_image_b64()
        prefixed = f"data:image/jpeg;base64,{raw_b64}"

        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   side_effect=Exception("SAM not available")), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="guide"):

            resp = client.post("/api/quiltify", json={
                "image_base64": prefixed,
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        assert resp.status_code == 200
        assert resp.json()["validation_errors"] == []

    def test_guide_fallback_on_ollama_error(self, client):
        """When Ollama fails, guide should be cutting instructions text."""
        image_b64 = _make_test_image_b64()

        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   side_effect=Exception("down")), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, side_effect=Exception("timeout")):

            resp = client.post("/api/quiltify", json={
                "image_base64": image_b64,
                "grid_width": 10,
                "grid_height": 10,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["guide"]) > 0
        assert "###" in data["guide"]


# ---------------------------------------------------------------------------
# /api/guide — regenerate guide from edited pattern
# ---------------------------------------------------------------------------

class TestGuideEndpoint:
    def test_regenerate_guide(self, client):
        """Guide endpoint should accept a pattern and return updated guide + SVGs."""
        pattern = _make_valid_pattern_payload()

        with patch("backend.routers.guide.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="# Updated Guide\n\nNew content here."):

            resp = client.post("/api/guide", json={
                "pattern": pattern,
                "title": "My Custom Quilt",
            })

        assert resp.status_code == 200
        data = resp.json()

        assert data["guide"] == "# Updated Guide\n\nNew content here."
        assert "<svg" in data["svg"]
        assert "<svg" in data["cutting_svg"]
        assert len(data["cutting_chart"]) > 0
        assert "pattern_json" in data

    def test_guide_with_ollama_down(self, client):
        """Guide endpoint should fall back to cutting instructions."""
        pattern = _make_valid_pattern_payload()

        with patch("backend.routers.guide.ollama_client.generate_guide",
                   new_callable=AsyncMock, side_effect=Exception("connection refused")):

            resp = client.post("/api/guide", json={"pattern": pattern})

        assert resp.status_code == 200
        data = resp.json()
        # Should contain cutting instruction text
        assert len(data["guide"]) > 0
        assert "###" in data["guide"]

    def test_guide_with_validation_errors(self, client):
        """Guide should still work even with validation errors in pattern."""
        pattern = _make_valid_pattern_payload()
        # Make the pattern invalid — block references non-existent fabric
        pattern["blocks"].append({
            "x": 0, "y": 0, "width": 1, "height": 1, "fabric_id": "f_nonexistent"
        })

        with patch("backend.routers.guide.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="guide"):

            resp = client.post("/api/guide", json={"pattern": pattern})

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["validation_errors"]) > 0


# ---------------------------------------------------------------------------
# /api/export/svg
# ---------------------------------------------------------------------------

class TestExportSvg:
    def test_export_svg_returns_svg_file(self, client):
        pattern = _make_valid_pattern_payload()
        resp = client.post("/api/export/svg", json={"pattern": pattern})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/svg+xml"
        assert "attachment" in resp.headers.get("content-disposition", "")
        assert "quilt-pattern.svg" in resp.headers["content-disposition"]
        body = resp.text
        assert "<svg" in body
        assert "</svg>" in body

    def test_export_svg_dimensions(self, client):
        """SVG should reflect the grid dimensions."""
        pattern = _make_valid_pattern_payload(grid_width=20, grid_height=15)
        resp = client.post("/api/export/svg", json={"pattern": pattern})
        assert resp.status_code == 200
        # SVG should contain rects — at least 2 blocks
        assert resp.text.count("<rect") >= 2


# ---------------------------------------------------------------------------
# /api/export/csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_export_csv_returns_csv_file(self, client):
        pattern = _make_valid_pattern_payload()
        resp = client.post("/api/export/csv", json={"pattern": pattern})

        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "cutting-chart.csv" in resp.headers.get("content-disposition", "")

        # Parse CSV
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)

        # Header row
        assert rows[0] == ["Fabric", "Color Hex", "Cut Width (in)", "Cut Height (in)", "Quantity"]
        # At least one data row
        assert len(rows) >= 2

    def test_csv_data_matches_pattern(self, client):
        """CSV fabric names should match the pattern's fabrics."""
        pattern = _make_valid_pattern_payload()
        resp = client.post("/api/export/csv", json={"pattern": pattern})
        reader = csv.reader(io.StringIO(resp.text))
        rows = list(reader)

        fabric_names_in_csv = {row[0] for row in rows[1:]}
        fabric_names_in_pattern = {f["name"] for f in pattern["fabrics"]}
        # Every CSV fabric should be from our pattern
        assert fabric_names_in_csv.issubset(fabric_names_in_pattern)


# ---------------------------------------------------------------------------
# /api/export/pdf
# ---------------------------------------------------------------------------

class TestExportPdf:
    def test_pdf_returns_501_without_weasyprint(self, client):
        """PDF export should return 501 when weasyprint is not installed."""
        pattern = _make_valid_pattern_payload()
        resp = client.post("/api/export/pdf", json={"pattern": pattern})
        # weasyprint is almost certainly not installed in the test env
        assert resp.status_code == 501
        assert "weasyprint" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Cross-pipeline integration: pattern consistency
# ---------------------------------------------------------------------------

class TestPatternConsistency:
    """Verify that patterns produced by /generate and /quiltify are internally
    consistent — every cell covered, no overlaps, fabrics referenced correctly."""

    def _assert_pattern_valid(self, pattern_json: dict):
        """Assert that a pattern_json is geometrically valid."""
        gw = pattern_json["grid_width"]
        gh = pattern_json["grid_height"]
        fab_ids = {f["id"] for f in pattern_json["fabrics"]}

        # All block fabric_ids reference known fabrics
        for b in pattern_json["blocks"]:
            assert b["fabric_id"] in fab_ids, f"Unknown fabric {b['fabric_id']}"

        # No overlaps, full coverage
        covered = set()
        for b in pattern_json["blocks"]:
            for dy in range(b["height"]):
                for dx in range(b["width"]):
                    cell = (b["x"] + dx, b["y"] + dy)
                    assert cell not in covered, f"Overlap at {cell}"
                    covered.add(cell)

        expected = {(x, y) for x in range(gw) for y in range(gh)}
        assert covered == expected, f"{len(expected - covered)} uncovered cells"

    def test_generate_pattern_valid(self, client):
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="g"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test", "grid_width": 15, "grid_height": 20, "palette_size": 4,
            })

        self._assert_pattern_valid(resp.json()["pattern_json"])

    def test_quiltify_pattern_valid(self, client):
        image_b64 = _make_striped_image_b64()

        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   side_effect=Exception("n/a")), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="g"):

            resp = client.post("/api/quiltify", json={
                "image_base64": image_b64,
                "grid_width": 12,
                "grid_height": 12,
                "palette_size": 3,
            })

        self._assert_pattern_valid(resp.json()["pattern_json"])

    def test_generate_cutting_chart_matches_blocks(self, client):
        """Total pieces in cutting chart should account for all blocks."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="g"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test", "grid_width": 10, "grid_height": 10, "palette_size": 3,
            })

        data = resp.json()
        chart_total = sum(p["quantity"] for p in data["cutting_chart"])
        block_count = len(data["pattern_json"]["blocks"])
        # Each block maps to exactly one cutting chart entry (possibly grouped)
        assert chart_total == block_count

    def test_fabric_areas_are_positive(self, client):
        """Every fabric used by blocks should have positive total_sqin."""
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="g"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            resp = client.post("/api/generate", json={
                "prompt": "test", "grid_width": 10, "grid_height": 10,
            })

        pj = resp.json()["pattern_json"]
        used_ids = {b["fabric_id"] for b in pj["blocks"]}
        for fab in pj["fabrics"]:
            if fab["id"] in used_ids:
                assert fab["total_sqin"] > 0, f"Fabric {fab['name']} has 0 sqin"


# ---------------------------------------------------------------------------
# Round-trip: generate → guide → export
# ---------------------------------------------------------------------------

class TestFullWorkflow:
    """Simulate user workflow: generate a pattern, regenerate guide, then export."""

    def test_generate_then_guide_then_export(self, client):
        # Step 1: Generate
        with patch("backend.routers.generate.flux_pipeline.generate_quilt_image",
                   return_value=None), \
             patch("backend.routers.generate.ollama_client.generate_block_layout",
                   new_callable=AsyncMock, side_effect=Exception("down")), \
             patch("backend.routers.generate.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Initial guide"), \
             patch("backend.routers.generate.flux_pipeline.pipeline_status",
                   return_value={"loaded": False, "type": "none"}):

            gen_resp = client.post("/api/generate", json={
                "prompt": "sunset quilt",
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        assert gen_resp.status_code == 200
        pattern_json = gen_resp.json()["pattern_json"]

        # Step 2: Regenerate guide with the same pattern
        with patch("backend.routers.guide.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Regenerated guide"):

            guide_resp = client.post("/api/guide", json={
                "pattern": pattern_json,
                "title": "My Sunset Quilt",
            })

        assert guide_resp.status_code == 200
        assert guide_resp.json()["guide"] == "Regenerated guide"

        # Pattern structure should be preserved through round-trip
        rt_pattern = guide_resp.json()["pattern_json"]
        assert rt_pattern["grid_width"] == pattern_json["grid_width"]
        assert rt_pattern["grid_height"] == pattern_json["grid_height"]
        assert len(rt_pattern["fabrics"]) == len(pattern_json["fabrics"])
        assert len(rt_pattern["blocks"]) == len(pattern_json["blocks"])

        # Step 3: Export as SVG using the same pattern
        svg_resp = client.post("/api/export/svg", json={"pattern": pattern_json})
        assert svg_resp.status_code == 200
        assert "<svg" in svg_resp.text

        # Step 4: Export as CSV
        csv_resp = client.post("/api/export/csv", json={"pattern": pattern_json})
        assert csv_resp.status_code == 200
        rows = list(csv.reader(io.StringIO(csv_resp.text)))
        assert len(rows) >= 2

    def test_quiltify_then_guide_then_export(self, client):
        """Full image upload workflow."""
        image_b64 = _make_striped_image_b64()

        # Step 1: Quiltify
        with patch("backend.routers.quiltify.quiltification.quiltify_image",
                   side_effect=Exception("n/a")), \
             patch("backend.routers.quiltify.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Guide from quiltify"):

            q_resp = client.post("/api/quiltify", json={
                "image_base64": image_b64,
                "grid_width": 10,
                "grid_height": 10,
                "palette_size": 3,
            })

        assert q_resp.status_code == 200
        pattern_json = q_resp.json()["pattern_json"]

        # Step 2: Re-guide
        with patch("backend.routers.guide.ollama_client.generate_guide",
                   new_callable=AsyncMock, return_value="Re-guide"):

            g_resp = client.post("/api/guide", json={"pattern": pattern_json})

        assert g_resp.status_code == 200

        # Step 3: Export CSV
        csv_resp = client.post("/api/export/csv", json={"pattern": pattern_json})
        assert csv_resp.status_code == 200
