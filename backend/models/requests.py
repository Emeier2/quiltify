from pydantic import BaseModel, Field
from typing import Optional
from .pattern import QuiltPatternSchema


class GenerateRequest(BaseModel):
    prompt: str
    grid_width: int = Field(default=40, ge=10, le=100)
    grid_height: int = Field(default=50, ge=10, le=100)
    palette_size: int = Field(default=6, ge=2, le=12)
    quilt_width_in: float = Field(default=60.0, ge=1.0, le=200.0)
    quilt_height_in: float = Field(default=72.0, ge=1.0, le=200.0)


class QuiltifyRequest(BaseModel):
    image_base64: str
    grid_width: int = Field(default=40, ge=10, le=100)
    grid_height: int = Field(default=50, ge=10, le=100)
    palette_size: int = Field(default=6, ge=2, le=12)
    quilt_width_in: float = Field(default=60.0, ge=1.0, le=200.0)
    quilt_height_in: float = Field(default=72.0, ge=1.0, le=200.0)


class GuideRequest(BaseModel):
    pattern: QuiltPatternSchema
    title: Optional[str] = None
