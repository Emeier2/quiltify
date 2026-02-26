from pydantic import BaseModel, Field
from typing import Optional


class BlockSchema(BaseModel):
    x: int
    y: int
    width: int
    height: int
    fabric_id: str


class FabricSchema(BaseModel):
    id: str
    color_hex: str
    name: str
    total_sqin: float = 0.0


class QuiltPatternSchema(BaseModel):
    grid_width: int = Field(default=40)
    grid_height: int = Field(default=50)
    block_size_in: float = Field(default=2.5)
    seam_allowance: float = Field(default=0.25)
    fabrics: list[FabricSchema] = []
    blocks: list[BlockSchema] = []


class CutPieceSchema(BaseModel):
    fabric_id: str
    fabric_name: str
    color_hex: str
    cut_width_in: float
    cut_height_in: float
    quantity: int


class CuttingChartSchema(BaseModel):
    block_size_in: float
    cut_size_in: float
    seam_allowance: float
    pieces: list[CutPieceSchema]
