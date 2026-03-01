# Quilt-Studio: Elizabeth Hartman-Style Upgrade Guide

## The Problem (Why It's Blocky)

The entire stack — from AI generation through data model to rendering — is locked to **axis-aligned rectangles only**. Every layer enforces this:

| Layer | File | Constraint |
|---|---|---|
| Data model | `grid_engine.py:38-44` | `Block` = `{x, y, width, height, fabric_id}` — rectangles only |
| Grid extraction | `grid_extractor.py:127-172` | Greedy rectangle merging — scans for maximal same-color rectangles |
| FLUX prompt | `flux_pipeline.py:25-29` | Hardcoded suffix: "solid fabric geometric **squares and rectangles**" |
| Ollama prompt | `prompts/json_layout.txt:2` | "no triangles, no curves, no half-square triangles" |
| Frontend canvas | `QuiltCanvas.tsx:66-81` | Only renders `<rect>` SVG elements |
| Cutting chart | `grid_engine.py:189-223` | Only computes rectangular cut pieces |

**Elizabeth Hartman's quilts are built on a rectangular grid, BUT each cell can contain a diagonal seam** — a half-square triangle (HST) or stitch-and-flip corner. This tiny addition (two-color cells with a diagonal split) is what transforms blocky pixel-art into recognizable animals and figures with contours, curves, and silhouettes.

Your system doesn't have this concept at all. Every cell is one solid color. That's why results look like Minecraft instead of Hartman.

---

## How Elizabeth Hartman Quilts Actually Work

Her technique is deceptively simple — the construction vocabulary has only ~5 elements:

1. **Solid square** — single fabric, single color
2. **HST (half-square triangle)** — square cut diagonally, two fabrics, two right triangles sharing a hypotenuse. Four rotations (NW/NE/SW/SE diagonal)
3. **Stitch-and-flip corner** — small square sewn diagonally across the corner of a larger rectangle, trimmed, creating a triangle inset. Four corner positions
4. **Flying geese** — 2:1 rectangle with two corner triangles, creating a center-pointing triangle. Four orientations
5. **Snowball block** — square with 1-4 corners flipped (stitch-and-flip on a square)

The key insight: **it's still a grid**. Every piece snaps to grid coordinates. The diagonals are the ONLY non-rectilinear element, and they always go corner-to-corner within a cell. This makes it computationally tractable — you're not doing arbitrary polygons, you're adding exactly one bit of information per cell: does it have a diagonal, and if so, which direction?

### The Cell State Alphabet

Each grid cell has one of these states:
```
SOLID        — one fabric fills the whole cell
HST_NWSE     — diagonal from NW to SE corner, two fabrics (fabric_a = upper-left triangle, fabric_b = lower-right)
HST_NESW     — diagonal from NE to SW corner, two fabrics (fabric_a = upper-right triangle, fabric_b = lower-left)
```

That's it. Three states (with two fabric assignments for HST states). Flying geese and stitch-and-flip are composed from these primitives across adjacent cells.

### Sizing Math

- Finished HST of size N: cut two squares at **(N + 7/8)"**
- Finished flying geese W x H: large rectangle **(W + 1/2)" x (H + 1/2)"**, small squares **(H + 1/2)"**
- Stitch-and-flip: rectangle at finished size + seam allowance, corner square at cell size + seam allowance

---

## The Fix: What Needs to Change (Layer by Layer)

### 1. Data Model (`grid_engine.py`)

Add a `cell_type` field to `Block` (or introduce a new `Cell` concept):

```python
class CellType(str, Enum):
    SOLID = "solid"
    HST_NWSE = "hst_nwse"  # diagonal: top-left to bottom-right
    HST_NESW = "hst_nesw"  # diagonal: top-right to bottom-left

@dataclass
class Block:
    x: int
    y: int
    width: int       # for HST blocks, always 1
    height: int      # for HST blocks, always 1
    fabric_id: str
    cell_type: CellType = CellType.SOLID
    fabric_id_b: Optional[str] = None  # second fabric for HST cells
```

**Decision point**: You could keep the greedy-merge optimization for SOLID blocks (multi-cell rectangles) and restrict HST to 1x1 cells only. This is how Hartman actually works — her HSTs are always single grid units.

The cutting chart needs new logic for HSTs: two triangles from one square cut, different from rectangular cuts.

### 2. Grid Extraction (`grid_extractor.py`)

This is the hardest part. Currently it does:
1. K-means quantization → flat color per cell
2. Greedy rectangle merging

For Hartman-style output, you need **edge-aware cell classification**:

1. K-means quantization (keep this)
2. For each cell, check if the original image has a **strong diagonal edge** passing through it
3. If yes: classify as HST_NWSE or HST_NESW, assign the two dominant colors on each side of the diagonal
4. If no: classify as SOLID with the dominant color
5. Then greedy-merge only the SOLID cells (HSTs stay 1x1)

**How to detect diagonals in a cell:**
```python
# For each cell (CELL_SAMPLE_PX x CELL_SAMPLE_PX patch):
# 1. Run Canny edge detection on the patch
# 2. Run Hough line detection
# 3. If a strong line runs approximately corner-to-corner (within ~15°), it's an HST candidate
# 4. Sample colors on each side of the diagonal to assign fabric_a and fabric_b

import cv2
patch = img_array[cy:cy+CELL_SAMPLE_PX, cx:cx+CELL_SAMPLE_PX]
edges = cv2.Canny(patch, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=CELL_SAMPLE_PX*0.6)
# Check if any line is approximately 45° diagonal
```

**Alternative approach — gradient-based:**
```python
# Compute the dominant gradient direction in the cell
# If gradient magnitude is high and direction is ~45° or ~135°, it's an HST
gx = cv2.Sobel(gray_patch, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray_patch, cv2.CV_64F, 0, 1, ksize=3)
angle = np.arctan2(gy.mean(), gx.mean()) * 180 / np.pi
```

### 3. AI Image Generation (`flux_pipeline.py`)

Change the FLUX style suffix to encourage diagonals:

```python
STYLE_SUFFIX = (
    ", pictorial modern quilt pattern in the style of Elizabeth Hartman, "
    "geometric patchwork using solid-color squares and half-square triangles, "
    "recognizable animal or object silhouette formed by diagonal seams and color blocking, "
    "bold solid fabric colors, clean geometric edges, quilt top view, "
    "no gradients no shading no textures no curves"
)
```

Consider adding **negative prompts** if using FLUX.1-dev with guidance:
- "blurry, photorealistic, gradient, smooth shading, watercolor, painterly"

**Better approach**: Instead of relying on FLUX to generate quilt-like images (it doesn't understand quilting construction), use the LLM to generate the **cell-by-cell layout directly** with HST support.

### 4. LLM Layout Generation (`prompts/json_layout.txt`)

This is where the biggest win is. Rewrite the prompt to include HST cells:

```
You are a quilt pattern designer specializing in Elizabeth Hartman-style pictorial quilts.
You build recognizable figures (animals, objects, nature scenes) from a grid of cells where
each cell is either a solid-color square or a half-square triangle (HST) with two colors.

HSTs create diagonal edges that form curves, contours, and silhouettes. Use them at the
boundaries of your subject to create smooth, recognizable shapes instead of blocky pixel art.

JSON FORMAT:
{
  "fabrics": [
    {"id": "f1", "color_hex": "#1b2d5b", "name": "Navy Solid"},
    {"id": "f2", "color_hex": "#f5f0dc", "name": "Cream"}
  ],
  "blocks": [
    {"x": 0, "y": 0, "width": 4, "height": 6, "fabric_id": "f1", "cell_type": "solid"},
    {"x": 4, "y": 3, "width": 1, "height": 1, "fabric_id": "f1", "cell_type": "hst_nwse", "fabric_id_b": "f2"}
  ]
}

CELL TYPES:
- "solid": single fabric fills the entire cell (can span multiple cells via width/height > 1)
- "hst_nwse": diagonal from top-left to bottom-right. fabric_id fills upper-left triangle,
  fabric_id_b fills lower-right triangle. MUST be 1×1.
- "hst_nesw": diagonal from top-right to bottom-left. fabric_id fills upper-right triangle,
  fabric_id_b fills lower-left triangle. MUST be 1×1.

DESIGN PRINCIPLES:
1. Use solid blocks for large color masses (background, body fill)
2. Use HSTs along the EDGES of your subject to create smooth contours
3. A row of HSTs in the same direction creates a diagonal line/slope
4. Alternating HST directions creates zigzag/sawtooth edges
5. Place HSTs where a real quilter would — at silhouette boundaries
6. The subject should be clearly recognizable as a specific animal/object
7. Study Elizabeth Hartman's style: bold, graphic, modern, whimsical
```

You'll also need new few-shot examples in `prompts/examples/` that demonstrate HST usage.

### 5. Frontend Rendering (`QuiltCanvas.tsx`)

Add SVG `<polygon>` rendering for HST cells:

```tsx
{pattern.blocks.map((block, i) => {
  const fab = fabricMap[block.fabric_id]
  const color = fab?.color_hex ?? '#cccccc'
  const px = block.x * CELL_PX
  const py = block.y * CELL_PX
  const pw = block.width * CELL_PX
  const ph = block.height * CELL_PX

  if (block.cell_type === 'hst_nwse') {
    const fabB = fabricMap[block.fabric_id_b!]
    const colorB = fabB?.color_hex ?? '#cccccc'
    return (
      <g key={i}>
        {/* Upper-left triangle */}
        <polygon points={`${px},${py} ${px+pw},${py} ${px},${py+ph}`} fill={color} stroke="#fff" strokeWidth={0.5} />
        {/* Lower-right triangle */}
        <polygon points={`${px+pw},${py} ${px+pw},${py+ph} ${px},${py+ph}`} fill={colorB} stroke="#fff" strokeWidth={0.5} />
      </g>
    )
  }
  if (block.cell_type === 'hst_nesw') {
    // Mirror diagonal
    const fabB = fabricMap[block.fabric_id_b!]
    const colorB = fabB?.color_hex ?? '#cccccc'
    return (
      <g key={i}>
        <polygon points={`${px},${py} ${px+pw},${py} ${px+pw},${py+ph}`} fill={color} stroke="#fff" strokeWidth={0.5} />
        <polygon points={`${px},${py} ${px+pw},${py+ph} ${px},${py+ph}`} fill={colorB} stroke="#fff" strokeWidth={0.5} />
      </g>
    )
  }
  // Default: solid rectangle
  return <rect key={i} x={px} y={py} width={pw} height={ph} fill={color} stroke="#fff" strokeWidth={0.5} />
})}
```

The paint/edit tool also needs updating — clicking an HST cell should cycle through: solid fabric A → solid fabric B → HST_NWSE → HST_NESW.

### 6. SVG Export (`svg_renderer.py`)

Same change as the frontend — render `<polygon>` for HST blocks instead of `<rect>`.

### 7. Cutting Chart (`grid_engine.py`)

HST cutting is different from rectangular cutting:

```python
# For HST blocks:
# Cut ONE square at (finished_size + 7/8)", then cut diagonally
# Each square yields TWO HST units
# So quantity of squares needed = ceil(hst_count / 2) for each fabric pair

hst_cut_size = block_size_in + 0.875  # 7/8 inch = 0.875
```

Group HSTs by fabric pair (both fabric_a and fabric_b matter), not just one fabric.

---

## Research Leads: Libraries and Papers

### Must-Read Papers

1. **"A Mathematical Foundation for Foundation Paper Pieceable Quilts"** — Leake, Bernstein, Davis, Agrawala (SIGGRAPH 2021)
   - Proves when a quilt pattern can be sewn via paper piecing using acyclic hypergraph theory
   - [ACM link](https://dl.acm.org/doi/10.1145/3450626.3459853)

2. **"Sketch-Based Design of Foundation Paper Pieceable Quilts"** — Leake, Bernstein, Agrawala (UIST 2022)
   - Sketch-completion algorithm that extends partial sketches into full planar meshes of pieceable polygon faces
   - Directly relevant for converting arbitrary outlines into quiltable geometry
   - [ACM link](https://dl.acm.org/doi/abs/10.1145/3526113.3545643)

3. **"Pixelated Image Abstraction"** — Gerstner, DeCarlo, Alexa (2012)
   - Optimizes over superpixels + constrained color palettes for pixel-art-style output
   - Good pre-processing step before grid extraction
   - [PDF](https://gfx.cs.princeton.edu/pubs/Gerstner_2012_PIA/Gerstner_2012_PIA_full.pdf)

### Most Actionable Libraries

| Library | What it does | Why you need it |
|---|---|---|
| **vtracer** (Rust/Python) | Raster → colored SVG polygons | Convert FLUX output to vector regions before grid snapping. `pip install vtracer`. [GitHub](https://github.com/visioncortex/vtracer) |
| **Shapely** (Python) | Polygon boolean ops, constrained Delaunay | Manipulate quilt regions as proper geometry. [Docs](https://shapely.readthedocs.io) |
| **DiffVG** (PyTorch) | Differentiable SVG rasterizer | Optimize SVG paths via gradient descent against a target image. [GitHub](https://github.com/BachiLi/diffvg) |
| **PyTorch-SVGRender** | Unified framework for neural SVG generation | Wraps DiffVG + VectorFusion + SVGDreamer. [GitHub](https://github.com/ximinng/PyTorch-SVGRender) |

### AI Models for Vector/Geometric Generation

| Model | Why it matters |
|---|---|
| **Chat2SVG** (CVPR 2025) | LLM generates SVG template using constrained primitives, then diffusion refines. Could constrain primitives to quilt-legal shapes only. [GitHub](https://github.com/kingnobro/Chat2SVG) |
| **SVGDreamer** (CVPR 2024) | Text-to-SVG via score distillation. Routes text tokens to different vector paths via cross-attention. [GitHub](https://github.com/ximinng/SVGDreamer) |
| **VectorFusion** (CVPR 2023) | Score distillation sampling from frozen diffusion model + DiffVG. Foundation paper for the field. [arXiv](https://arxiv.org/abs/2211.11319) |
| **SAMVG** (2023) | SAM segmentation → vector paths. Already close to your quiltification pipeline. [arXiv](https://arxiv.org/html/2311.05276v2) |

### Existing Quilt Code (Reference Only)

- [rhjones/quilt-pattern-generator](https://github.com/rhjones/quilt-pattern-generator) — JS, generates random HST + square patterns
- Emily Xie's **Interwoven** (LACMA 2023) — p5.js generative patchwork art, closest existing generative art to Hartman style. [Coverage](https://nftnow.com/features/emily-xie-and-lacma-present-interwoven-inspired-by-quilts-and-generative-art/)

### Tessellation and Tiling

- **Tactile / TactileJS** — All 93 isohedral tiling types. [C++](https://github.com/isohedral/tactile) / [JS](https://github.com/isohedral/tactile-js)
- **Tessagon** — Python tessellation on 2D manifolds. [GitHub](https://github.com/cwant/tessagon)
- Craig Kaplan's book: *Introductory Tiling Theory for Computer Graphics* (Morgan & Claypool)

---

## Recommended Implementation Order

### Phase 1: Add HST to the Data Model + Renderer (quickest visual win)
1. Add `cell_type` and `fabric_id_b` to `Block` dataclass
2. Update `QuiltCanvas.tsx` to render `<polygon>` for HST blocks
3. Update `svg_renderer.py` for SVG export
4. Update TypeScript types (`types/pattern.ts`) and Pydantic schemas (`models/pattern.py`)
5. Manually create a test pattern JSON with HSTs to verify rendering

### Phase 2: LLM-Direct Layout with HSTs (biggest quality jump)
1. Rewrite `prompts/json_layout.txt` to include HST cell types
2. Create 3-4 new few-shot examples with HSTs (bird, cat, fox, tree)
3. Update `ollama_client.py` to validate the new schema
4. Test with qwen2.5:32b — this model should handle it

### Phase 3: Image-to-Pattern with Edge Detection
1. Add Canny/Sobel edge detection per cell in `grid_extractor.py`
2. Classify cells as SOLID vs HST based on diagonal edge presence
3. Update the greedy merge to skip HST cells
4. Test with the quiltify (image upload) path

### Phase 4: Cutting Chart + Guide Updates
1. Update `grid_engine.py` cutting chart for HST pieces
2. Update `cutting_calculator.py` with HST sizing math
3. Update `guide_writing.txt` prompt to describe HST assembly
4. Update cutting diagram SVG to show diagonal cuts

### Phase 5 (Stretch): Advanced AI Pipeline
1. Replace FLUX pixel output → Chat2SVG or direct LLM SVG generation
2. Use vtracer to vectorize reference images before grid extraction
3. Experiment with DiffVG optimization: start from LLM layout, optimize colors/positions against a reference image via differentiable rendering

---

## Quick Reference: HST Geometry

```
HST_NWSE (diagonal: top-left → bottom-right)
┌─────────┐
│╲  fab_a  │
│  ╲       │
│    ╲     │
│ fab_b ╲  │
│         ╲│
└─────────┘

Wait, let me be precise:

HST_NWSE: line from (0,0) to (1,1)
┌─────────┐
│╲        │
│  ╲  B   │
│ A  ╲    │
│      ╲  │
│        ╲│
└─────────┘
fabric_id (A) = lower-left triangle
fabric_id_b (B) = upper-right triangle

HST_NESW: line from (1,0) to (0,1)
┌─────────┐
│        ╱│
│  A   ╱  │
│    ╱    │
│  ╱  B   │
│╱        │
└─────────┘
fabric_id (A) = upper-left triangle
fabric_id_b (B) = lower-right triangle
```

Pick a convention and stick with it across all layers. Document it in a constant.

---

## Key Insight

You don't need to throw away the grid system. Hartman's genius is that **she works on a grid** — she just allows cells to have diagonal splits. Your entire architecture (grid coordinates, cutting charts, block merging for solid regions) can stay. You're adding one enum field and one optional fabric reference to each block. The visual improvement will be dramatic for very little structural change.

The LLM layout path (Phase 2) will give you the biggest quality jump with the least code. A well-prompted qwen2.5:32b with good few-shot examples can absolutely produce Hartman-style layouts with HSTs — the JSON schema is simple enough.

---

## Implementation Progress Log

### 2026-03-01 — Phase 0 + Phase 1 (partial)

**Status: Phase 0 COMPLETE, Phase 1 COMPLETE, Phase 2 COMPLETE, Phase 3 COMPLETE, Phase 4 COMPLETE**

All work is unstaged on the `master` branch. 169/169 tests passing.

#### Phase 0 — Model/Schema Reset (DONE)
Implemented by Codex, verified and test-fixed in this session.

- `grid_engine.py` — `Block.corners` dict, `cell_sizes` row-major array, variable-cell `column_widths()`/`row_heights()`/`block_dimensions_in()`, `piece_type` in cutting chart, corner-aware `compute_fabric_areas()` with 50/50 split, validation for column/row consistency + corner fabric refs
- `models/pattern.py` — Pydantic schemas: `corners`, `cell_sizes`, `quilt_width_in`/`quilt_height_in`, `piece_type`
- `models/requests.py` — Replaced `block_size_inches` with `quilt_width_in`/`quilt_height_in`
- `types/pattern.ts` — TypeScript interfaces updated
- `QuiltCanvas.tsx` — Variable-cell offset rendering, `renderCornerTriangles()` with `<polygon>`, `subBlockCorners()` preserves corners on block splits, merge logic respects corners
- `svg_renderer.py` — Variable-cell offsets + corner triangle polygons (svgwrite + fallback)
- `cutting_calculator.py` — Labels "base rectangles" vs "corner squares"
- All routers updated for new schema
- All frontend pages/components updated

#### Phase 1 — AI Geometry Map (DONE)
- `geometry_layout.txt` — New prompt: variable cell sizes + stitch-and-flip corners, 10 rules
- `geometry_critic.txt` — Validator/repair prompt for critic model
- `ollama_client.py` — Dual-model setup (qwen2.5:14b + gemma3:12b critic), `_chat_with_model()`, `_load_examples()` includes `cell_sizes` in assistant output, critique/repair pass, geometry examples loaded as few-shot
- 3 new few-shot examples with corners (all pass validation):
  - `geometry_example_cat.json` — 8x10 grid, 4 corners (ear tips + chin rounding)
  - `geometry_example_tree.json` — 8x12 grid, 10 corners (sloped branch edges)
  - `geometry_example_heart.json` — 8x8 grid, 12 corners (curved tapering)
- `guide_writing.txt` — Updated with stitch-and-flip assembly instructions (5-step technique)

#### Test Fixes Applied
- `test_prompt_contains_grid_dimensions` — Read `user_message` from kwargs
- `test_dimensions_correct` — Accept float dimensions from variable-cell offsets
- `test_contains_fabric_names` / `test_contains_dimensions_label` — Graceful when svgwrite missing
- `test_passes_validate` (3 layout examples) — Synthesize `cell_sizes` for old-format examples

#### Phase 2 — Image-to-Pattern Edge Detection (DONE)
- `grid_extractor.py` — Added diagonal color boundary analysis:
  - `_CORNER_EDGE_THRESHOLD = 0.35` — minimum fraction of secondary-color pixels for corner classification
  - `_detect_corners()` — For each cell, checks two possible diagonals (TR-BL and TL-BR). Compares mean color of each half against quantized palette centers. If halves have different dominant colors and the non-primary half meets threshold, classifies as stitch-and-flip corner.
  - `_check_diagonal()` — Helper that analyzes one diagonal split. Computes per-pixel distances to primary vs secondary cluster centers, requires ≥35% of pixels closer to secondary.
  - `_merge_grid_to_blocks()` — Now accepts `corner_map`, emits corner cells as 1×1 blocks with `corners` dict first, then greedy-merges remaining solid cells.
  - `extract_pattern_from_image()` — Wired `_detect_corners` → `_merge_grid_to_blocks` pipeline.
- `test_grid_extractor.py` — 9 new tests:
  - `TestDetectCorners` (5): solid cells, axis-aligned boundaries, NW diagonal, SE diagonal, rejection of mostly-wrong-color cells
  - `TestMergeWithCorners` (4): normal merge without corners, corner stays 1×1, adjacent cells still merge, full coverage check

#### Phase 3 — Cutting Chart + Guide Updates (DONE)
- `cutting_calculator.py` — Added `CORNER_WASTE_FACTOR = 1.60` (~50% trim + 10% cutting waste) for stitch-and-flip corner squares. `calculate_requirements()` now applies per-piece waste factors (corner vs base). Cutting sequence labels corner squares as "stitch-and-flip".
- `svg_renderer.py` — Cutting diagram now draws dashed diagonal line across corner square pieces + "S&F" label to visually indicate stitch-and-flip.
- `test_grid_engine.py` — 7 new `TestCornerCuttingChart` tests: corner squares in chart, correct fabric assignment, square shape, correct size (cell + 2*SA), quantity matches corner count, both piece types present, 50/50 area split.
- `test_cutting_calculator.py` — 4 new `TestCornerWasteFactor` tests: waste factor constant, corner pieces increase yardage, cutting sequence labels corners, both piece types in sequence.

#### Phase 4 — Advanced AI Pipeline (DONE — vtracer integrated)

Research findings:
- **Chat2SVG** (CVPR 2025): Research-only, needs SDXL+SAM+diffvg+CUDA, Linux only, multi-minute generation. NOT practical.
- **DiffVG** (SIGGRAPH 2020): Continuous parameter optimization, wrong fit for discrete block placement. Needs C++ build + CUDA. NOT practical.
- **vtracer** (Rust/PyO3): `pip install vtracer`, native speed, no GPU, works everywhere. **Integrated.**

Implementation:
- `vtracer_service.py` — New service wrapping vtracer for image vectorization. Three functions:
  - `vectorize_image(image_bytes)` → SVG string. Parameters tuned for quilt images: `color_precision=5`, `filter_speckle=8`, `layer_difference=24`, `mode="polygon"` (geometric, not organic).
  - `vectorize_and_rasterize(image_bytes, width, height)` → cleaned JPEG bytes via cairosvg round-trip.
  - `clean_for_extraction(image_bytes, grid_width, grid_height)` → pipeline entry point, produces cleaned image at exact grid resolution.
- `grid_extractor.py` — Now attempts vtracer cleaning before resize/quantization. Graceful fallback: if vtracer or cairosvg unavailable, uses original image (zero behavior change).
- `test_vtracer_service.py` — 9 tests: availability check, solid/two-color vectorization, SVG structure validation, fill colors, polygon mode, graceful fallbacks, grid extractor integration.
