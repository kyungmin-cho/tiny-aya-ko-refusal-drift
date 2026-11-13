#!/bin/bash
# E4 (first run: T100/C100/T200/C200 arms) on one CUDA GPU: deps -> pinned model/data download -> orchestrator.
# Usage:  HF_TOKEN=<your token> bash scripts/run_e4.sh   (token read from the environment only, never printed)
set -euo pipefail
cd "$(dirname "$0")/.."
R=experiments/e4
mkdir -p "$R/logs"
uv sync --frozen > "$R/logs/uv_sync.log" 2>&1
uv run python -I scripts/e4_pod_download.py > "$R/logs/download.log" 2>&1
grep -q ALLDONE "$R/logs/download.log" || { echo "download failed: see $R/logs/download.log"; exit 1; }
nohup uv run python -I scripts/e4_orchestrate.py "$R" > "$R/logs/orchestrate.log" 2>&1 &
echo "orchestrator pid $!  (tail -f $R/logs/orchestrate.log)"
