from pydantic import BaseModel, Field
from typing import Optional


class BlockSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int
    fabric_id: str
    corners: Optional[dict[str, str]] = None


class FabricSchema(BaseModel):
    id: str
    color_hex: str
    name: str
    total_sqin: float = 0.0


class QuiltPatternSchema(BaseModel):
    grid_width: int = Field(default=40)
    grid_height: int = Field(default=50)
    quilt_width_in: float = Field(default=60.0)
    quilt_height_in: float = Field(default=72.0)
    seam_allowance: float = Field(default=0.25)
    fabrics: list[FabricSchema] = []
    blocks: list[BlockSchema] = []
    # Row-major: length = grid_width * grid_height
    cell_sizes: list[dict[str, float]] = []


class CutPieceSchema(BaseModel):
    fabric_id: str
    fabric_name: str
    color_hex: str
    cut_width_in: float
    cut_height_in: float
    quantity: int
    piece_type: str = Field(default="base")


class CuttingChartSchema(BaseModel):
    block_size_in: float
    cut_size_in: float
    seam_allowance: float
    pieces: list[CutPieceSchema]
