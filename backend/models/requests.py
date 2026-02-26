from pydantic import BaseModel, Field
from typing import Optional
from .pattern import QuiltPatternSchema


class GenerateRequest(BaseModel):
    prompt: str
    grid_width: int = Field(default=40, ge=10, le=100)
    grid_height: int = Field(default=50, ge=10, le=100)
    palette_size: int = Field(default=6, ge=2, le=12)
    block_size_inches: float = Field(default=2.5, ge=1.0, le=6.0)


class QuiltifyRequest(BaseModel):
    image_base64: str
    grid_width: int = Field(default=40, ge=10, le=100)
    grid_height: int = Field(default=50, ge=10, le=100)
    palette_size: int = Field(default=6, ge=2, le=12)
    block_size_inches: float = Field(default=2.5, ge=1.0, le=6.0)


class GuideRequest(BaseModel):
    pattern: QuiltPatternSchema
    title: Optional[str] = None
