#!/bin/bash

set -euo pipefail

echo "--- Phase 1: Running Mathematical Baseline ---"
uv run ./degroot.py

echo "--- Phase 2: Extracting LLM Data ---" 
uv run ./data_extraction.py

echo "--- Phase 3: Merging LLM and DeGroot Data ---" 
uv run ./data_grouping.py

echo "--- Phase 4: Generating Plots ---" 
uv run ./plotting.py