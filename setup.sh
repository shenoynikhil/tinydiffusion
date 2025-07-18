#!/bin/bash

# Create a new uv project for tinydiffusion
echo "Setting up tinydiffusion project with uv..."

# Install dependencies from pyproject.toml
echo "Installing dependencies..."
uv sync

# Install optional dev dependencies
echo "Installing development dependencies..."
uv sync --extra dev

# Activate env
echo "Setup complete! Activate env with: source .venv/bin/activate"
source .venv/bin/activate
