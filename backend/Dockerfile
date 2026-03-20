# Backend Dockerfile
# Uses PyTorch CUDA base — requires NVIDIA GPU on the host.
# To run CPU-only, change FROM to python:3.11-slim and
# comment out the GPU-heavy lines in requirements.txt.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System libraries needed by OpenCV and WeasyPrint
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libffi-dev \
        libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy only the backend package (keeps the image lean)
COPY backend/ backend/

EXPOSE 8000

# Run from project root so relative imports resolve correctly
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
