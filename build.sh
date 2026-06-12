#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies using Render's native uv
uv pip install -r requirements.txt
