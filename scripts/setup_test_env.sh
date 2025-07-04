#!/usr/bin/env bash
# Lightweight setup script for local testing.
# Installs dependencies and the package in editable mode.
set -e
pip install -r requirements.txt
pip install -e .
