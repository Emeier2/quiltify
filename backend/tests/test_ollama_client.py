"""Unit tests for ollama_client.py — Ollama API wrapper for guide + layout generation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
import httpx

from backend.services.ollama_client import (
    _load_prompt,
    _load_examples,
    _build_guide_user_message,
    _chat,
    generate_guide,
    generate_block_layout,
    check_health,
    PROMPTS_DIR,
    EXAMPLES_DIR,
)
from backend.services.grid_engine import QuiltPattern, Fabric, Block


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _sample_pattern_json() -> dict:
    return {
        "grid_width": 40,
        "grid_height": 50,
        "quilt_width_in": 100.0,
        "quilt_height_in": 125.0,
        "seam_allowance": 0.25,
        "finished_width_in": 100.0,
        "finished_height_in": 125.0,
        "fabrics": [
            {"id": "f1", "color_hex": "#1b2d5b", "name": "Kona Cotton - Navy",
             "total_sqin": 2880.0, "fat_quarters": 8},
            {"id": "f2", "color_hex": "#f5f0dc", "name": "Kona Cotton - Cream",
             "total_sqin": 2880.0, "fat_quarters": 8},
        ],
        "blocks": [
            {"x": 0, "y": 0, "width": 40, "height": 25, "fabric_id": "f1"},
            {"x": 0, "y": 25, "width": 40, "height": 25, "fabric_id": "f2"},
        ],
        "cell_sizes": [{"w": 2.5, "h": 2.5} for _ in range(2000)],
    }


def _sample_cutting_instructions() -> list[str]:
    return [
        "Kona Cotton - Navy: Cut 1 piece 120.0\" × 75.0\"",
        "Kona Cotton - Cream: Cut 1 piece 120.0\" × 75.0\"",
    ]


def _mock_chat_response(content: str) -> httpx.Response:
    """Build a fake httpx.Response with Ollama chat format."""
    return httpx.Response(
        status_code=200,
        json={"message": {"role": "assistant", "content": content}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPrompt:
    def test_loads_existing_file(self):
        text = _load_prompt("guide_writing.txt")
        assert len(text) > 0
        assert "quilting" in text.lower()

    def test_loads_layout_prompt(self):
        text = _load_prompt("json_layout.txt")
        assert len(text) > 0
        assert "JSON" in text

    def test_missing_file_returns_empty(self):
        text = _load_prompt("nonexistent_prompt_file.txt")
        assert text == ""


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _build_guide_user_message
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildGuideUserMessage:
    def test_contains_title(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), "My Owl Quilt"
        )
        assert "My Owl Quilt" in msg

    def test_default_title(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), None
        )
        assert "Modern Geometric Quilt" in msg

    def test_contains_dimensions(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), None
        )
        assert "100.0" in msg
        assert "125.0" in msg

    def test_contains_fabric_names(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), None
        )
        assert "Kona Cotton - Navy" in msg
        assert "Kona Cotton - Cream" in msg

    def test_contains_cutting_instructions(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), None
        )
        assert 'Cut 1 piece 120.0"' in msg

    def test_contains_block_count(self):
        pj = _sample_pattern_json()
        msg = _build_guide_user_message(pj, _sample_cutting_instructions(), None)
        assert f"TOTAL BLOCKS: {len(pj['blocks'])}" in msg

    def test_contains_seam_allowance(self):
        msg = _build_guide_user_message(
            _sample_pattern_json(), _sample_cutting_instructions(), None
        )
        assert "0.25" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _chat
# ─────────────────────────────────────────────────────────────────────────────

class TestChat:
    @pytest.mark.asyncio
    async def test_sends_correct_payload(self):
        captured = {}

        async def mock_post(url, json=None, **kwargs):
            captured["url"] = url
            captured["json"] = json
            return _mock_chat_response("Hello quilter!")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _chat("system prompt", "user message")

        assert result == "Hello quilter!"
        assert captured["json"]["model"] is not None
        assert captured["json"]["messages"][0]["role"] == "system"
        assert captured["json"]["messages"][1]["role"] == "user"
        assert captured["json"]["stream"] is False

    @pytest.mark.asyncio
    async def test_returns_content_from_response(self):
        async def mock_post(url, json=None, **kwargs):
            return _mock_chat_response("## Overview\nThis is a beautiful quilt.")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await _chat("sys", "usr")

        assert "Overview" in result
        assert "beautiful quilt" in result

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        async def mock_post(url, json=None, **kwargs):
            resp = httpx.Response(status_code=500, text="Internal Server Error",
                                 request=httpx.Request("POST", url))
            return resp

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(httpx.HTTPStatusError):
                await _chat("sys", "usr")


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_guide
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateGuide:
    @pytest.mark.asyncio
    async def test_returns_guide_text(self):
        guide_text = "## Overview\nA 100\" × 125\" quilt in Navy and Cream."

        with patch("backend.services.ollama_client._chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = guide_text
            result = await generate_guide(
                _sample_pattern_json(), _sample_cutting_instructions(), "Test Quilt"
            )

        assert result == guide_text
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args
        assert "Test Quilt" in call_args[0][1]  # user_message contains title

    @pytest.mark.asyncio
    async def test_uses_file_prompt_when_available(self):
        with patch("backend.services.ollama_client._chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "guide text"
            await generate_guide(_sample_pattern_json(), _sample_cutting_instructions())

        system_prompt = mock_chat.call_args[0][0]
        # Should load from guide_writing.txt (which exists)
        assert "quilting" in system_prompt.lower()

    @pytest.mark.asyncio
    async def test_falls_back_to_default_prompt(self):
        with patch("backend.services.ollama_client._load_prompt", return_value=""):
            with patch("backend.services.ollama_client._chat", new_callable=AsyncMock) as mock_chat:
                mock_chat.return_value = "guide text"
                await generate_guide(_sample_pattern_json(), _sample_cutting_instructions())

        system_prompt = mock_chat.call_args[0][0]
        assert "quilting instructor" in system_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_block_layout
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateBlockLayout:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self):
        layout = {
            "fabrics": [{"id": "f1", "color_hex": "#1b2d5b", "name": "Navy"}],
            "blocks": [{"x": 0, "y": 0, "width": 40, "height": 50, "fabric_id": "f1"}],
        }
        raw_response = json.dumps(layout)

        with patch("backend.services.ollama_client._chat_with_model", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = [raw_response, raw_response]
            result = await generate_block_layout("owl quilt", 40, 50, 1, 100.0, 125.0)

        assert result == layout
        assert result["fabrics"][0]["id"] == "f1"

    @pytest.mark.asyncio
    async def test_extracts_json_from_prose(self):
        """Ollama sometimes wraps JSON in explanation text."""
        layout = {"fabrics": [{"id": "f1", "color_hex": "#fff", "name": "White"}], "blocks": []}
        raw_response = f"Here is your layout:\n{json.dumps(layout)}\nHope that helps!"

        with patch("backend.services.ollama_client._chat_with_model", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = [raw_response, raw_response]
            result = await generate_block_layout("test", 10, 10, 1, 25.0, 25.0)

        assert result["fabrics"][0]["id"] == "f1"

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_invalid_json(self):
        with patch("backend.services.ollama_client._chat_with_model", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = ["I can't generate that pattern, sorry.", ""]
            result = await generate_block_layout("test", 10, 10, 1, 25.0, 25.0)

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_empty_response(self):
        with patch("backend.services.ollama_client._chat_with_model", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = ["", ""]
            result = await generate_block_layout("test", 10, 10, 1, 25.0, 25.0)

        assert result == {}

    @pytest.mark.asyncio
    async def test_prompt_contains_grid_dimensions(self):
        with patch("backend.services.ollama_client._chat_with_model", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = ["{}", "{}"]
            await generate_block_layout("sunflower", 30, 40, 5, 75.0, 100.0)

        user_msg = mock_chat.call_args_list[0].kwargs.get(
            "user_message", mock_chat.call_args_list[0][0][2] if len(mock_chat.call_args_list[0][0]) > 2 else ""
        )
        assert "30" in user_msg
        assert "40" in user_msg
        assert "5" in user_msg
        assert "sunflower" in user_msg


# ─────────────────────────────────────────────────────────────────────────────
# Tests: check_health
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_returns_true_when_reachable(self):
        async def mock_get(url, **kwargs):
            return httpx.Response(status_code=200, json={"models": []})

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            assert await check_health() is True

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self):
        async def mock_get(url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            assert await check_health() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self):
        async def mock_get(url, **kwargs):
            raise httpx.TimeoutException("Timed out")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            assert await check_health() is False

    @pytest.mark.asyncio
    async def test_returns_false_on_non_200(self):
        async def mock_get(url, **kwargs):
            return httpx.Response(status_code=503, text="Service Unavailable")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.get = mock_get
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            assert await check_health() is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _load_examples
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadExamples:
    def test_layout_examples_load(self):
        messages = _load_examples("layout_example*.json")
        # 3 layout files × 2 messages each = 6
        assert len(messages) >= 6
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_guide_example_loads(self):
        messages = _load_examples("guide_example*.json")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "## Overview" in messages[1]["content"]

    def test_nonexistent_pattern_returns_empty(self):
        messages = _load_examples("no_such_file_*.json")
        assert messages == []

    def test_layout_example_assistant_is_valid_json(self):
        messages = _load_examples("layout_example*.json")
        for msg in messages:
            if msg["role"] == "assistant":
                data = json.loads(msg["content"])
                assert "fabrics" in data
                assert "blocks" in data


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Layout example validation (QuiltPattern.validate)
# ─────────────────────────────────────────────────────────────────────────────

class TestLayoutExampleValidation:
    @pytest.fixture(params=["layout_example_bird.json",
                            "layout_example_mountain.json",
                            "layout_example_flower.json"])
    def example_data(self, request):
        path = EXAMPLES_DIR / request.param
        return json.loads(path.read_text(encoding="utf-8"))

    def test_passes_validate(self, example_data):
        gw = example_data["grid_width"]
        gh = example_data["grid_height"]
        qw = example_data.get("quilt_width_in", 60.0)
        qh = example_data.get("quilt_height_in", 72.0)
        cell_sizes = example_data.get("cell_sizes", [
            {"w": qw / gw, "h": qh / gh} for _ in range(gw * gh)
        ])
        pattern = QuiltPattern(
            grid_width=gw,
            grid_height=gh,
            quilt_width_in=qw,
            quilt_height_in=qh,
            cell_sizes=cell_sizes,
        )
        for f in example_data["fabrics"]:
            pattern.fabrics.append(Fabric(id=f["id"], color_hex=f["color_hex"], name=f["name"]))
        for b in example_data["blocks"]:
            pattern.blocks.append(Block(x=b["x"], y=b["y"], width=b["width"],
                                        height=b["height"], fabric_id=b["fabric_id"],
                                        corners=b.get("corners") or {}))
        errors = pattern.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_uses_all_declared_fabrics(self, example_data):
        declared = {f["id"] for f in example_data["fabrics"]}
        used = {b["fabric_id"] for b in example_data["blocks"]}
        assert used == declared


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Guide example content
# ─────────────────────────────────────────────────────────────────────────────

class TestGuideExampleContent:
    @pytest.fixture
    def guide_data(self):
        path = EXAMPLES_DIR / "guide_example.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_contains_all_seven_section_headers(self, guide_data):
        output = guide_data["output"]
        required_headers = [
            "## Overview",
            "## Materials & Fabric Requirements",
            "## Cutting Instructions",
            "## Block Assembly",
            "## Row Assembly",
            "## Quilt Top Assembly",
            "## Finishing Notes",
        ]
        for header in required_headers:
            assert header in output, f"Missing header: {header}"

    def test_input_matches_user_message_format(self, guide_data):
        inp = guide_data["input"]
        assert "TITLE:" in inp
        assert "FINISHED SIZE:" in inp
        assert "CUTTING INSTRUCTIONS" in inp


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _chat with extra_messages
# ─────────────────────────────────────────────────────────────────────────────

class TestChatExtraMessages:
    @pytest.mark.asyncio
    async def test_extra_messages_inserted_between_system_and_user(self):
        captured = {}

        async def mock_post(url, json=None, **kwargs):
            captured["json"] = json
            return _mock_chat_response("ok")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            extras = [
                {"role": "user", "content": "example input"},
                {"role": "assistant", "content": "example output"},
            ]
            await _chat("sys prompt", "real question", extra_messages=extras)

        msgs = captured["json"]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "example input"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "example output"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "real question"
        assert len(msgs) == 4

    @pytest.mark.asyncio
    async def test_no_extra_messages_preserves_original_behavior(self):
        captured = {}

        async def mock_post(url, json=None, **kwargs):
            captured["json"] = json
            return _mock_chat_response("ok")

        with patch("backend.services.ollama_client.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await _chat("sys", "usr")

        msgs = captured["json"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
