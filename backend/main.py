"""Quiltify — FastAPI application entry point."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import generate, quiltify, guide, export
from .services import ollama_client, flux_pipeline, svg_generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup checks
    ollama_ok = await ollama_client.check_health()
    if ollama_ok:
        logger.info("Ollama is reachable")
    else:
        logger.warning("Ollama not reachable at startup — guide generation will be limited")

    pipe_status = flux_pipeline.pipeline_status()
    logger.info(f"FLUX pipeline status: {pipe_status}")

    sv_status = svg_generator.generator_status()
    logger.info(f"StarVector status: {sv_status}")

    yield
    # Shutdown (nothing to clean up currently)


app = FastAPI(
    title="Quilt Studio",
    description="AI-powered pictorial modern quilt pattern generator",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — defaults cover local dev; set ALLOWED_ORIGINS in production
_default_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://localhost",
    "http://localhost:80",
]
_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
allow_origins = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    if _env_origins
    else _default_origins
)
logger.info(f"CORS allowed origins: {allow_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(generate.router)
app.include_router(quiltify.router)
app.include_router(guide.router)
app.include_router(export.router)


@app.get("/health")
async def health() -> dict:
    ollama_ok = await ollama_client.check_health()
    pipe = flux_pipeline.pipeline_status()
    sv = svg_generator.generator_status()
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "flux_pipeline": pipe,
        "starvector": sv,
    }


@app.get("/")
async def root() -> dict:
    return {"message": "Quilt Studio API", "docs": "/docs"}
