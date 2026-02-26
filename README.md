# Quiltify

AI-powered pictorial modern quilt pattern generator.

Generates pictorial modern quilt patterns from text prompts or photos, and produces accurate cutting guides (cutting charts, block assembly steps, yardage calculations).

## Architecture

```
TEXT PROMPT  →  FLUX.1-dev  →  Grid Extractor  →  QuiltPattern  →  Ollama Guide
                               (K-means + cell                    (qwen2.5:14b)
                                sampling + block
                                merge)

IMAGE INPUT  →  SAM + ControlNet  →  Grid Extractor  →  QuiltPattern  →  Ollama Guide
                (FLUX Canny)
```

## Project Structure

```
quilt-studio/
  backend/
    main.py                 # FastAPI entry point
    routers/                # generate, quiltify, guide, export
    services/
      grid_engine.py        # Core domain model (Block, Fabric, QuiltPattern, CuttingChart)
      cutting_calculator.py # Fat quarter math
      svg_renderer.py       # SVG grid preview + cutting diagrams
      flux_pipeline.py      # FLUX.1-dev image generation
      grid_extractor.py     # K-means + cell sampling → QuiltPattern
      quiltification.py     # SAM + ControlNet img2img
      ollama_client.py      # Ollama API wrapper
      color_matcher.py      # Kona Cotton palette matching (CIELAB)
    data/
      kona_cotton_palette.json  # ~160 Kona Cotton colors
    tests/
      test_grid_engine.py   # Unit tests
  frontend/
    src/
      components/           # QuiltCanvas, FabricPalette, CuttingChart, GuideViewer, ...
      pages/                # Generate, Quiltify, Export
      api/client.ts         # Typed fetch wrapper
      types/pattern.ts      # TypeScript types
```

## Setup

### Backend

```bash
cd backend

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
# For full AI generation (GPU required):
pip install -r requirements.txt

# Lightweight install (no image generation, grid engine + guide only):
pip install fastapi uvicorn pydantic httpx python-multipart svgwrite scikit-learn Pillow numpy

# Pull the Ollama model for guide generation:
ollama pull qwen2.5:14b

# (Optional) Download SAM checkpoint for quiltification:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P ~/.cache/sam/

# Run
uvicorn backend.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

### HuggingFace Token (for FLUX.1-dev)

FLUX.1-dev requires accepting the license and a HF token:
1. Accept at https://huggingface.co/black-forest-labs/FLUX.1-dev
2. `export HF_TOKEN=hf_...`

If no token is provided, the app falls back to FLUX.1-schnell (no auth, slightly lower quality).

## Model Requirements

| Model | VRAM | Notes |
|---|---|---|
| FLUX.1-dev (Q4) | ~9 GB | Primary generation; requires HF token |
| FLUX.1-schnell | ~8 GB | Fallback; no auth required |
| ControlNet Canny | +1.5 GB | Quiltification only |
| SAM vit_b | ~0.5 GB | Quiltification only |
| qwen2.5:14b (Ollama) | ~10 GB RAM | Guide writing; runs on CPU |

Total peak VRAM (text path): ~9–11 GB. Ollama uses system RAM, not VRAM.

## Running Without a GPU

The app degrades gracefully:
- No FLUX pipeline → uses a synthetic striped placeholder pattern
- No Ollama → returns cutting instructions as plain text
- All grid engine logic, SVG rendering, and export work without any AI dependencies

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/generate` | Text prompt → pattern + guide |
| POST | `/api/quiltify` | Image → quilt pattern + guide |
| POST | `/api/guide` | Regenerate guide from edited pattern |
| POST | `/api/export/svg` | Download pattern as SVG |
| POST | `/api/export/csv` | Download cutting chart as CSV |
| POST | `/api/export/pdf` | Download full guide as PDF |
| GET | `/health` | Service health check |

## Running Tests

```bash
cd Development/quilt-studio
python -m pytest backend/tests/ -v
```

The grid engine tests run with no GPU or Ollama dependency.

## Design Notes

The core tension in AI quilting tools: AI images are beautiful but not "buildable."

This app resolves it by using AI purely for visual design, then extracting a structured grid via K-means color quantization + greedy rectangle merging. The grid engine computes all measurements mathematically — the LLM only writes prose around pre-computed numbers, never inventing its own measurements.

Every cut size in the guide traces directly to `block_size_in + 2 × seam_allowance`.
