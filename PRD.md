# Quiltify — Product Requirements Document

> **Purpose**: Persistent context document for handoff between sessions, LLMs (Claude, Ollama, etc.), and collaborators. Updated: 2026-02-22.

---

## 1. Product Overview

**Quiltify** (project directory: `quilt-studio`) is an AI-powered web application that generates pictorial modern quilt patterns from text prompts or photos, then produces accurate, buildable cutting guides (cutting charts, block assembly steps, yardage calculations).

### Core Design Principle

> AI images are beautiful but not "buildable." This app resolves the tension by using AI purely for visual design, then extracting a structured grid via K-means color quantization + greedy rectangle merging. The grid engine computes all measurements mathematically — the LLM only writes prose around pre-computed numbers, never inventing measurements.

### Target Users

Quilters (beginner to intermediate) who want to create modern pictorial quilt patterns without needing to manually design grids or calculate cutting dimensions.

---

## 2. Architecture

```
TEXT PROMPT  →  FLUX.1-dev  →  Grid Extractor  →  QuiltPattern  →  Ollama Guide
                               (K-means + cell                    (qwen2.5:32b)
                                sampling + block
                                merge)

IMAGE INPUT  →  SAM + ControlNet  →  Grid Extractor  →  QuiltPattern  →  Ollama Guide
                (FLUX Canny)
```

### Tech Stack

| Layer     | Technology                                    |
|-----------|-----------------------------------------------|
| Frontend  | React 18 + TypeScript, Vite, inline CSS       |
| Backend   | Python 3.11+, FastAPI, Pydantic               |
| AI/ML     | FLUX.1-dev (image gen), SAM (segmentation), ControlNet Canny (img2img) |
| LLM       | Ollama with qwen2.5:32b (guide writing)       |
| Color     | Kona Cotton palette (~160 colors), CIELAB matching |
| Export    | SVG (svgwrite), CSV, PDF (weasyprint)         |

### Graceful Degradation

The app works without any AI dependencies:
- No FLUX pipeline → Ollama layout generation (confidence 0.5) → synthetic striped placeholder (confidence 0.0)
- No Ollama → returns cutting instructions as plain text
- All grid engine logic, SVG rendering, and export work without GPU or LLM

---

## 3. Project Structure

```
quilt-studio/
  backend/
    main.py                    # FastAPI entry point, CORS, lifespan startup checks
    models/
      pattern.py               # QuiltPatternSchema, BlockSchema, FabricSchema, etc.
      requests.py              # GenerateRequest, QuiltifyRequest, GuideRequest
      guide.py                 # GuideSection, QuiltingGuide (NOT WIRED — see Known Gaps)
    routers/
      generate.py              # POST /api/generate — text prompt → pattern
      quiltify.py              # POST /api/quiltify — image → pattern
      guide.py                 # POST /api/guide — regenerate guide from edited pattern
      export.py                # POST /api/export/{svg,csv,pdf}
    services/
      grid_engine.py           # Core domain: Fabric, Block, CutPiece, CuttingChart, QuiltPattern
      grid_extractor.py        # K-means + greedy rectangle merge → QuiltPattern
      color_matcher.py         # CIELAB nearest-match to Kona Cotton palette
      cutting_calculator.py    # Yardage math, fat quarter calculations, cutting sequence
      svg_renderer.py          # Grid preview SVG + cutting diagram SVG
      flux_pipeline.py         # FLUX.1-dev / schnell text-to-image
      quiltification.py        # SAM + ControlNet Canny img2img
      ollama_client.py         # Ollama /api/chat wrapper (guide + layout generation)
    prompts/
      guide_writing.txt        # System prompt for Ollama guide generation
      json_layout.txt          # System prompt for Ollama block layout generation
    data/
      kona_cotton_palette.json # ~160 Kona Cotton colors with hex/RGB/CIELAB
    tests/
      test_grid_engine.py      # 26 unit tests (all passing)
      test_color_matcher.py    # 16 unit tests (all passing)
      test_cutting_calculator.py # 11 unit tests (all passing)
      test_svg_renderer.py     # 15 unit tests (all passing)
  frontend/
    src/
      App.tsx                  # Tab navigation (Generate / Quiltify / Export), top-level state
      main.tsx                 # React entry point
      api/client.ts            # Typed fetch wrapper for all API endpoints
      types/pattern.ts         # TypeScript interfaces matching backend schemas
      pages/
        Generate.tsx           # Full text→pattern→edit→export workflow
        Quiltify.tsx           # Full image→pattern→edit→export workflow
        Export.tsx             # Standalone export page (wired via onSendToExport callback)
      components/
        QuiltCanvas.tsx        # Interactive SVG grid editor (paint cells, split/merge blocks)
        FabricPalette.tsx      # Fabric selector + inline rename
        CuttingChart.tsx       # Grouped per-fabric cut table
        GuideViewer.tsx        # Collapsible-section markdown guide renderer
        PromptInput.tsx        # Prompt box + example chips + advanced options
        ImageUpload.tsx        # Drag-and-drop / file picker
        ConfidenceScore.tsx    # Color-coded confidence badge
    package.json               # Dependencies (react, vite)
    vite.config.ts
    tsconfig.json
  README.md
```

---

## 4. API Endpoints

| Method | Path              | Description                        | Status       |
|--------|-------------------|------------------------------------|--------------|
| POST   | `/api/generate`   | Text prompt → pattern + guide      | Complete     |
| POST   | `/api/quiltify`   | Image → quilt pattern + guide      | Complete     |
| POST   | `/api/guide`      | Regenerate guide from edited pattern | Complete   |
| POST   | `/api/export/svg` | Download pattern as SVG            | Complete     |
| POST   | `/api/export/csv` | Download cutting chart as CSV      | Complete     |
| POST   | `/api/export/pdf` | Download full guide as PDF         | Complete     |
| GET    | `/health`         | Service health check               | Complete     |

### Request/Response Schemas

**GenerateRequest**: `{ prompt, grid_width (10-100, default 40), grid_height (10-100, default 50), palette_size (2-12, default 6), block_size_inches (1.0-6.0, default 2.5) }`

**QuiltifyRequest**: `{ image_base64, grid_width, grid_height, palette_size, block_size_inches }` (same defaults)

**GuideRequest**: `{ pattern: QuiltPatternSchema, title?: string }`

**GenerateResponse**: `{ pattern_json, svg, cutting_svg, cutting_chart, guide, confidence_score, validation_errors, pipeline_status, image_b64 }`

**QuiltifyResponse**: `{ pattern_json, svg, cutting_svg, cutting_chart, guide, confidence_score, validation_errors, original_image_b64, quilt_image_b64 }`

---

## 5. Core Domain Model (`grid_engine.py`)

```
Fabric       { id, color_hex, name, total_sqin } → fat_quarters(), yardage()
Block        { x, y, width, height, fabric_id }  → cells(), area_cells()
CutPiece     { fabric_id, fabric_name, color_hex, cut_width_in, cut_height_in, quantity }
CuttingChart { block_size_in, seam_allowance, pieces[] } → by_fabric(), total_pieces()
QuiltPattern { grid_width, grid_height, block_size_in, seam_allowance, fabrics[], blocks[] }
             → validate(), to_cutting_chart(), to_dict(), from_dict()
```

**Key formula**: `cut_size_in = block_size_in + 2 * seam_allowance`

**Validation checks**: out-of-bounds blocks, fabric ID references, cell overlaps, uncovered cells.

---

## 6. Frontend Component Map

```
App.tsx
├── GeneratePage
│   ├── PromptInput (8 example prompts, collapsible advanced options)
│   ├── ConfidenceScore (green/yellow/red badge)
│   ├── QuiltCanvas (interactive SVG: paint cells, split blocks T/B/L/R, horizontal merge)
│   ├── FabricPalette (click-to-select, double-click-to-rename)
│   ├── CuttingChart (grouped by fabric, sorted by area)
│   └── GuideViewer (collapsible accordion, markdown rendering)
├── QuiltifyPage
│   ├── ImageUpload (drag-and-drop / file picker)
│   ├── ConfidenceScore
│   ├── QuiltCanvas
│   ├── FabricPalette
│   ├── CuttingChart
│   └── GuideViewer
└── ExportPage (pattern: QuiltPatternSchema | null)
    └── Export buttons (SVG / CSV / PDF) — wired via onSendToExport callback
```

**Navigation**: `useState<Page>` tab switching (not react-router).

---

## 7. What is COMPLETE and Working

### Backend (fully implemented)
- All 4 routers wired end-to-end
- `grid_engine.py` — mature domain model with validation, serialization, cutting chart math
- `grid_extractor.py` — full K-means + greedy-rect pipeline; graceful fallback to synthetic stripes
- `color_matcher.py` — CIELAB implementation + built-in fallback palette
- `cutting_calculator.py` — full yardage/fat-quarter/WOF math
- `svg_renderer.py` — grid SVG and cutting diagram SVG; fallback for missing svgwrite
- `flux_pipeline.py` — full FLUX.1-dev Q4 → schnell fallback chain
- `quiltification.py` — full SAM + ControlNet pipeline with Canny fallback paths
- `ollama_client.py` — `generate_guide()` and `generate_block_layout()` both implemented
- `export.py` — SVG and CSV complete; PDF via weasyprint with full HTML template
- 126 unit tests — all passing (all 7 service modules + grid_extractor fallback)

### Frontend (fully implemented)
- All 3 pages built with async loading states, error handling, recalculate flows
- `QuiltCanvas` — interactive editing: paint cells, split blocks, horizontal merge
- `FabricPalette` — click-to-select + double-click-to-rename
- All other components fully functional
- Typed API client covering all endpoints

---

## 8. Known Gaps and Bugs (Priority Order)

### ~~Bug 1: Export page never receives a pattern~~ FIXED (2026-02-22)
`App.tsx` now passes `onSendToExport` callback to both `GeneratePage` and `QuiltifyPage`. Clicking "Send to Export" sets the pattern and navigates to the Export tab.

### ~~Gap 2: `cutting_svg` returned by API but never displayed~~ FIXED (2026-02-22)
Added "Cutting Diagram" tab to both `Generate.tsx` and `Quiltify.tsx`. Renders `result.cutting_svg` via `dangerouslySetInnerHTML` (safe — server-generated SVG from own backend).

### Gap 3: Server-rendered `svg` field stored but never displayed
**Problem**: The `result.svg` (server-rendered grid SVG) is returned and stored in state but never rendered. The interactive `QuiltCanvas` re-renders client-side, making this redundant — but it could serve as a static preview or print view.

### ~~Gap 4: `generate_block_layout()` is built but never called~~ FIXED (2026-02-22)
`generate.py` now tries `ollama_client.generate_block_layout()` as an intermediate fallback (confidence 0.5) when FLUX is unavailable. Falls through to synthetic stripes (confidence 0.0) if Ollama layout fails validation.

### Gap 5: `QuiltingGuide` model defined but not wired
**File**: `models/guide.py`
**Problem**: The structured `QuiltingGuide` Pydantic model (with `sections: list[GuideSection]`) is never used. The guide router returns raw `str`. The `GuideViewer` frontend parses markdown headings itself. This was likely a planned structured-guide API that was abandoned for raw text passthrough.

### ~~Gap 6: Missing tests for 7 of 8 service modules~~ FIXED (2026-02-25)
All 8 service modules now have test coverage. 126 tests, all passing.
- `grid_engine.py` — 26 tests
- `color_matcher.py` — 16 tests
- `cutting_calculator.py` — 11 tests
- `svg_renderer.py` — 15 tests
- `ollama_client.py` — 25 tests (mocked httpx, pytest-asyncio)
- `flux_pipeline.py` — 16 tests (mocked diffusers/torch, fallback chain)
- `quiltification.py` — 17 tests (mocked SAM/ControlNet/cv2, all 4 edge detection paths)

### ~~Gap 7: `react-router-dom` declared but unused~~ FIXED (2026-02-22)
Removed `react-router-dom` from `frontend/package.json`.

### ~~Gap 8: `node_modules` not installed~~ FIXED (2026-02-22)
Dependencies installed via `npm install`.

---

## 9. Model Requirements

| Model             | VRAM     | Notes                              |
|--------------------|----------|------------------------------------|
| FLUX.1-dev (Q4)   | ~9 GB    | Primary generation; requires HF token |
| FLUX.1-schnell    | ~8 GB    | Fallback; no auth required         |
| ControlNet Canny  | +1.5 GB  | Quiltification only                |
| SAM vit_b         | ~0.5 GB  | Quiltification only                |
| qwen2.5:32b       | ~20 GB RAM | Guide writing + layout generation via Ollama; CPU |

Total peak VRAM (text path): ~9-11 GB. Ollama uses system RAM, not VRAM.

### HuggingFace Token (FLUX.1-dev)

1. Accept license at `https://huggingface.co/black-forest-labs/FLUX.1-dev`
2. `export HF_TOKEN=hf_...`
3. Without token, falls back to FLUX.1-schnell

---

## 10. Setup & Run

### Backend
```bash
cd Development/quilt-studio/backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Or lightweight (no GPU): pip install fastapi uvicorn pydantic httpx python-multipart svgwrite scikit-learn Pillow numpy
ollama pull qwen2.5:32b
cd ..  # back to quilt-studio/
uvicorn backend.main:app --reload
```

### Frontend
```bash
cd Development/quilt-studio/frontend
npm install
npm run dev
```
Open http://localhost:5173

### Tests
```bash
cd Development/quilt-studio
python -m pytest backend/tests/ -v
```

---

## 11. Suggested Next Steps (for any LLM or developer picking this up)

1. **Wire the `QuiltingGuide` structured model** (Gap 5) — replace raw text passthrough with structured sections
2. **Render the server-side `svg` field** (Gap 3) — useful as a static print preview
4. Consider migrating inline styles to CSS modules or Tailwind for maintainability
5. Consider adding a pattern gallery / save/load functionality
6. Consider adding undo/redo to the QuiltCanvas editor

---

## 12. File Modification History

All files were created in a single build session on **February 20, 2025** between 22:05-23:00. The last-touched files (23:00) were: `main.py`, `export.py`, `package.json`, `index.html`, `App.tsx`, `Generate.tsx`, `Quiltify.tsx`, `README.md`. The session likely ended while wiring the Export page — `setExportPattern` was added to App.tsx state but the callback was never passed down.

**February 22, 2026**: Bug fix + feature session. Fixed Export page wiring (Bug 1), added Cutting Diagram tab (Gap 2), wired Ollama layout fallback (Gap 4), added 42 new tests for color_matcher/cutting_calculator/svg_renderer (Gap 6 partial), removed unused react-router-dom (Gap 7), installed deps (Gap 8), updated Ollama model default from qwen2.5:14b to qwen2.5:32b.

**February 25, 2026**: Test coverage completion. Added 58 new tests for the 3 remaining untested service modules: `ollama_client.py` (25 tests), `flux_pipeline.py` (16 tests), `quiltification.py` (17 tests). All use mocked GPU/network dependencies. Total: 126 tests, all passing. Installed pytest-asyncio. Gap 6 fully closed.

---

## 13. Key Conventions

- **Backend**: Pydantic models in `models/`, business logic in `services/`, route handlers in `routers/`
- **Frontend**: Pages in `pages/`, reusable components in `components/`, API layer in `api/`
- **Domain model**: Pure Python dataclasses in `grid_engine.py`, no ORM
- **Error handling**: Try/except with graceful fallback + logging; never crash the pipeline
- **Inline CSS**: Frontend uses React inline styles (no CSS files, no Tailwind)
- **Color palette**: Deep purple (#2a1040, #4a2060) + warm cream (#faf8f5) + serif fonts (Georgia)
