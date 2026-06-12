#!/usr/bin/env bash
# exit on error
set -o errexit

# Install dependencies into Render's managed venv (the same
# interpreter the start command runs with)
python -m pip install -r requirements.txt
