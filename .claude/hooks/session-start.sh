#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Installing backend dependencies (lightweight)..."
pip install fastapi uvicorn pydantic httpx python-multipart svgwrite scikit-learn Pillow numpy pytest

echo "Installing frontend dependencies..."
cd "$CLAUDE_PROJECT_DIR/frontend"
npm install

echo "Session start setup complete."
