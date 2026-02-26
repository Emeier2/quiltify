"""
Ollama Client — wraps the local Ollama API for guide generation.

Ollama is expected to be running at http://localhost:11434 with
qwen2.5:32b pulled: `ollama pull qwen2.5:32b`
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

OLLAMA_BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:32b")
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


async def generate_guide(
    pattern_json: dict,
    cutting_instructions: list[str],
    title: str | None = None,
) -> str:
    """
    Call Ollama to generate a full prose quilting guide.
    Returns the raw text response.
    """
    system_prompt = _load_prompt("guide_writing.txt")
    if not system_prompt:
        system_prompt = _default_guide_system_prompt()

    user_message = _build_guide_user_message(pattern_json, cutting_instructions, title)

    return await _chat(system_prompt, user_message)


async def generate_block_layout(
    prompt: str,
    grid_width: int,
    grid_height: int,
    palette_size: int,
) -> dict:
    """
    Ask Ollama to suggest a JSON block layout based on a text prompt.
    Used as a fallback when FLUX is unavailable.
    """
    system_prompt = _load_prompt("json_layout.txt")
    if not system_prompt:
        system_prompt = _default_layout_system_prompt()

    user_message = (
        f"Create a quilt pattern JSON for: '{prompt}'\n"
        f"Grid: {grid_width} wide × {grid_height} tall\n"
        f"Use exactly {palette_size} fabrics.\n"
        f"Return ONLY valid JSON, no explanation."
    )

    raw = await _chat(system_prompt, user_message)
    # Extract JSON from response
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return {}


async def _chat(system_prompt: str, user_message: str, timeout: float = 120.0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


async def check_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_BASE}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def _build_guide_user_message(
    pattern_json: dict,
    cutting_instructions: list[str],
    title: str | None,
) -> str:
    finished_w = pattern_json.get("finished_width_in", "?")
    finished_h = pattern_json.get("finished_height_in", "?")
    block_sz = pattern_json.get("block_size_in", 2.5)
    sa = pattern_json.get("seam_allowance", 0.25)
    fabrics = pattern_json.get("fabrics", [])
    blocks = pattern_json.get("blocks", [])

    fabric_lines = "\n".join(
        f"  - {f['name']} ({f['color_hex']}): {f.get('total_sqin', 0):.0f} sq in, "
        f"{f.get('fat_quarters', 1)} fat quarter(s)"
        for f in fabrics
    )
    cutting_text = "\n".join(cutting_instructions)

    return f"""Write a complete quilting guide for this quilt:

TITLE: {title or 'Modern Geometric Quilt'}
FINISHED SIZE: {finished_w}" × {finished_h}"
FINISHED BLOCK SIZE: {block_sz}"
SEAM ALLOWANCE: {sa}"
TOTAL BLOCKS: {len(blocks)}

FABRICS:
{fabric_lines}

CUTTING INSTRUCTIONS (these numbers are exact — use them verbatim):
{cutting_text}

Write the guide now. Use every number from the cutting instructions exactly as given."""


def _default_guide_system_prompt() -> str:
    return """You are an expert quilting instructor writing a clear, accurate quilting guide for pictorial modern quilts.

RULES — follow these exactly:
1. Every measurement you write MUST come from the data provided. Do not invent or estimate any number.
2. Write in a warm, encouraging tone suitable for intermediate quilters.
3. Structure the guide with these sections:
   ## Overview
   ## Materials & Fabric Requirements
   ## Cutting Instructions
   ## Block Assembly
   ## Row Assembly
   ## Quilt Top Assembly
   ## Finishing Notes
4. In Cutting Instructions, reproduce the exact cut sizes from the data.
5. In Block Assembly, describe how pieces join to form blocks.
6. Keep instructions concise but complete — a quilter should be able to follow them at the cutting table."""


def _default_layout_system_prompt() -> str:
    return """You are a quilt pattern designer. When asked, you output ONLY valid JSON describing a quilt block layout.

The JSON format:
{
  "fabrics": [
    {"id": "f1", "color_hex": "#hex", "name": "Kona Cotton - ColorName"}
  ],
  "blocks": [
    {"x": 0, "y": 0, "width": 2, "height": 3, "fabric_id": "f1"}
  ]
}

Rules:
- Blocks must not overlap
- Blocks must cover every cell in the grid
- Use simple rectangular blocks (no triangles, curves, or HSTs)
- Design the pattern to suggest the subject (animal, plant, landscape) using solid-color rectangles
- Each fabric id must be referenced in at least one block"""
