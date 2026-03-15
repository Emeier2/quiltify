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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_MODEL_GEOMETRY = os.environ.get("OLLAMA_MODEL_GEOMETRY", "qwen2.5:14b")
OLLAMA_MODEL_CRITIC = os.environ.get("OLLAMA_MODEL_CRITIC", "gemma3:12b")
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
EXAMPLES_DIR = PROMPTS_DIR / "examples"


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _load_examples(pattern: str) -> list[dict]:
    """Load JSON example files matching *pattern* from the examples directory.

    Each JSON file must have ``"prompt"``/``"input"`` and an expected output
    field.  Returns a list of ``{"role": "user", ...}`` / ``{"role":
    "assistant", ...}`` message pairs suitable for few-shot prompting.
    """
    messages: list[dict] = []
    if not EXAMPLES_DIR.is_dir():
        return messages
    for path in sorted(EXAMPLES_DIR.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Layout examples: user=prompt+grid spec, assistant=JSON output
        if "prompt" in data and "blocks" in data:
            qw = data.get("quilt_width_in", "")
            qh = data.get("quilt_height_in", "")
            user_content = (
                f"Create a quilt pattern JSON for: '{data['prompt']}'\n"
                f"Grid: {data['grid_width']} wide x {data['grid_height']} tall\n"
            )
            if qw and qh:
                user_content += f"Quilt size: {qw}\" wide x {qh}\" tall\n"
            user_content += (
                f"Use exactly {len(data['fabrics'])} fabrics.\n"
                f"Return ONLY valid JSON, no explanation."
            )
            output_dict: dict = {"fabrics": data["fabrics"], "blocks": data["blocks"]}
            if "cell_sizes" in data:
                output_dict["cell_sizes"] = data["cell_sizes"]
            assistant_content = json.dumps(output_dict, separators=(",", ":"))
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": assistant_content})
        # Guide examples: explicit input/output
        elif "input" in data and "output" in data:
            messages.append({"role": "user", "content": data["input"]})
            messages.append({"role": "assistant", "content": data["output"]})
    return messages


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
    examples = _load_examples("guide_example*.json")

    return await _chat(system_prompt, user_message, extra_messages=examples)


async def generate_block_layout(
    prompt: str,
    grid_width: int,
    grid_height: int,
    palette_size: int,
    quilt_width_in: float,
    quilt_height_in: float,
) -> dict:
    """
    Ask Ollama to suggest a JSON block layout with per-cell sizes and corners.
    Used as a fallback when FLUX is unavailable.
    """
    system_prompt = _load_prompt("geometry_layout.txt")
    if not system_prompt:
        system_prompt = _default_layout_system_prompt()

    user_message = (
        f"Create a quilt pattern JSON for: '{prompt}'\n"
        f"Grid: {grid_width} wide x {grid_height} tall\n"
        f"Quilt size: {quilt_width_in}\" wide x {quilt_height_in}\" tall\n"
        f"Use exactly {palette_size} fabrics.\n"
        f"CRITICAL: Blocks must cover ALL {grid_width * grid_height} cells with NO gaps. "
        f"Use large background blocks to fill remaining space. "
        f"Use stitch-and-flip corners (nw/ne/sw/se) to create diagonal details on planets.\n"
        f"Do NOT include cell_sizes — they will be computed automatically.\n"
        f"Return ONLY valid JSON with fabrics and blocks arrays, no explanation. /no_think"
    )
    examples = _load_examples("geometry_example*.json")

    raw = await _chat_with_model(
        model=OLLAMA_MODEL_GEOMETRY,
        system_prompt=system_prompt,
        user_message=user_message,
        extra_messages=examples,
    )
    layout = _extract_json(raw)
    if not layout:
        return {}

    # Post-process: fill gaps and fix overlaps via rasterize-then-merge
    layout = _postprocess_layout(layout, grid_width, grid_height)

    # Auto-fill cell_sizes if the model omitted them (uniform grid)
    if "cell_sizes" not in layout:
        cell_w = round(quilt_width_in / grid_width, 4)
        cell_h = round(quilt_height_in / grid_height, 4)
        layout["cell_sizes"] = [{"w": cell_w, "h": cell_h}
                                for _ in range(grid_width * grid_height)]

    # Critique/repair pass with smaller model
    critic_prompt = _load_prompt("geometry_critic.txt")
    if critic_prompt:
        # Strip cell_sizes before sending to critic to save tokens
        critic_layout = {k: v for k, v in layout.items() if k != "cell_sizes"}
        critic_message = (
            "Validate and repair this quilt JSON. Fix any errors and return ONLY JSON.\n"
            "Do NOT include cell_sizes in your output.\n"
            f"Grid: {grid_width}x{grid_height}\n"
            f"Quilt size: {quilt_width_in}x{quilt_height_in} inches\n"
            f"Input JSON:\n{json.dumps(critic_layout)}"
        )
        repaired_raw = await _chat_with_model(
            model=OLLAMA_MODEL_CRITIC,
            system_prompt=critic_prompt,
            user_message=critic_message,
        )
        repaired = _extract_json(repaired_raw)
        if repaired:
            repaired = _postprocess_layout(repaired, grid_width, grid_height)
            # Re-fill cell_sizes after critic pass
            if "cell_sizes" not in repaired:
                repaired["cell_sizes"] = layout["cell_sizes"]
            return repaired

    return layout


async def _chat(
    system_prompt: str,
    user_message: str,
    timeout: float = 120.0,
    extra_messages: list[dict] | None = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 8192},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            if resp.status_code != 200:
                return await _chat_via_generate(client, payload, timeout)
            data = resp.json()
            return data["message"]["content"]
        except Exception:
            return await _chat_via_generate(client, payload, timeout)


async def _chat_with_model(
    model: str,
    system_prompt: str,
    user_message: str,
    timeout: float = 120.0,
    extra_messages: list[dict] | None = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 8192},
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(f"{OLLAMA_BASE}/api/chat", json=payload)
            if resp.status_code != 200:
                return await _chat_via_generate(client, payload, timeout)
            data = resp.json()
            return data["message"]["content"]
        except Exception:
            return await _chat_via_generate(client, payload, timeout)


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
    {"x": 0, "y": 0, "width": 2, "height": 3, "fabric_id": "f1", "corners": {}}
  ],
  "cell_sizes": [{"w": 1.0, "h": 1.0}]
}

Rules:
- Blocks must not overlap
- Blocks must cover every cell in the grid
- Stitch-and-flip corners may be specified per block in "corners"
- Design the pattern to suggest the subject (animal, plant, landscape) using solid-color rectangles
- Each fabric id must be referenced in at least one block"""


def _postprocess_layout(
    layout: dict, grid_width: int, grid_height: int
) -> dict:
    """Rasterize model blocks to a grid and re-merge to fix overlaps/gaps.

    Steps:
    1. Paint each block onto a 2D grid (later blocks overwrite earlier ones).
       Preserve corner info for cells that have corners.
    2. Fill any uncovered cells with the background fabric (first fabric).
    3. Greedy-merge same-fabric cells into maximal rectangles.
    4. Re-attach corners to the merged blocks.
    """
    fabrics = layout.get("fabrics", [])
    blocks = layout.get("blocks", [])
    if not fabrics or not blocks:
        return layout

    bg_fabric = fabrics[0]["id"]  # first fabric is background

    # Build per-cell fabric_id grid and corner map
    grid = [[bg_fabric] * grid_width for _ in range(grid_height)]
    corner_cells: dict[tuple[int, int], dict[str, str]] = {}

    for blk in blocks:
        bx, by = blk.get("x", 0), blk.get("y", 0)
        bw, bh = blk.get("width", 1), blk.get("height", 1)
        fid = blk.get("fabric_id", bg_fabric)
        corners = blk.get("corners", {})
        for dy in range(bh):
            for dx in range(bw):
                cx, cy = bx + dx, by + dy
                if 0 <= cx < grid_width and 0 <= cy < grid_height:
                    grid[cy][cx] = fid
        # Attach corners to specific cells at actual block corners
        if corners:
            corner_positions = {
                "nw": (bx, by),
                "ne": (bx + bw - 1, by),
                "sw": (bx, by + bh - 1),
                "se": (bx + bw - 1, by + bh - 1),
            }
            for cname, cfid in corners.items():
                pos = corner_positions.get(cname)
                if pos and 0 <= pos[0] < grid_width and 0 <= pos[1] < grid_height:
                    corner_cells.setdefault(pos, {})[cname] = cfid

    # Greedy merge: corner cells become 1x1, others merge into maximal rects
    visited = [[False] * grid_width for _ in range(grid_height)]
    merged_blocks: list[dict] = []

    # First pass: emit corner cells as 1x1
    for (cx, cy), corners in corner_cells.items():
        visited[cy][cx] = True
        merged_blocks.append({
            "x": cx, "y": cy, "width": 1, "height": 1,
            "fabric_id": grid[cy][cx], "corners": corners,
        })

    # Second pass: greedy merge
    for gy in range(grid_height):
        for gx in range(grid_width):
            if visited[gy][gx]:
                continue
            fid = grid[gy][gx]
            max_w = 0
            while (gx + max_w < grid_width
                   and grid[gy][gx + max_w] == fid
                   and not visited[gy][gx + max_w]):
                max_w += 1
            max_h = 1
            while gy + max_h < grid_height:
                row_ok = all(
                    grid[gy + max_h][gx + dx] == fid
                    and not visited[gy + max_h][gx + dx]
                    for dx in range(max_w)
                )
                if not row_ok:
                    break
                max_h += 1
            for dy in range(max_h):
                for dx in range(max_w):
                    visited[gy + dy][gx + dx] = True
            merged_blocks.append({
                "x": gx, "y": gy, "width": max_w, "height": max_h,
                "fabric_id": fid, "corners": {},
            })

    layout["blocks"] = merged_blocks
    return layout


def _extract_json(raw: str) -> dict:
    import re
    # Strip <think>...</think> blocks (qwen3 reasoning mode)
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except (json.JSONDecodeError, ValueError):
        return {}


async def _chat_via_generate(
    client: httpx.AsyncClient,
    chat_payload: dict,
    timeout: float,
) -> str:
    """Fallback for Ollama servers that don't support /api/chat."""
    # Flatten messages into a single prompt
    messages = chat_payload.get("messages", [])
    prompt_lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt_lines.append(f"[{role.upper()}]\n{content}")
    prompt = "\n\n".join(prompt_lines)

    gen_payload = {
        "model": chat_payload.get("model"),
        "prompt": prompt,
        "stream": False,
        "options": chat_payload.get("options", {}),
    }
    resp = await client.post(f"{OLLAMA_BASE}/api/generate", json=gen_payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")
