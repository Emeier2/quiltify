"""Quiltify — FastAPI application entry point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import generate, quiltify, guide, export
from .services import ollama_client, flux_pipeline

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

    yield
    # Shutdown (nothing to clean up currently)


app = FastAPI(
    title="Quilt Studio",
    description="AI-powered Elizabeth Hartman-style quilt pattern generator",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
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
    return {
        "status": "ok",
        "ollama": ollama_ok,
        "flux_pipeline": pipe,
    }


@app.get("/")
async def root() -> dict:
    return {"message": "Quilt Studio API", "docs": "/docs"}
