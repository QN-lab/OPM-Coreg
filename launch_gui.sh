#!/usr/bin/env bash

# Exit on error
set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating..."
    python3 -m venv venv

    source venv/bin/activate

    echo "Installing package (from pyproject.toml)..."
    pip install --upgrade pip
    pip install -e .
else
    echo "Virtual environment found."
    source venv/bin/activate
fi

echo "Launching the GUI..."
opm-coreg-gui