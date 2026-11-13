#!/bin/bash
# E5 end-to-end on one CUDA GPU (we used a single A40 48GB): deps -> pinned model/data download -> orchestrator.
# Usage (from anywhere):  HF_TOKEN=<your token> bash experiments/e5/scripts/run_e5.sh
# HF_TOKEN is needed only for the gated CohereLabs/tiny-aya-* repos. It is read from the environment and never printed.
# Prerequisite: experiments/e1e3/*.jsonl and experiments/e5/data/eval_items/*.jsonl rebuilt with scripts/rebuild_eval_items.py (sha256-checked).
set -euo pipefail
cd "$(dirname "$0")/../../.."
R=experiments/e5
mkdir -p "$R/logs"
uv sync --frozen > "$R/logs/uv_sync.log" 2>&1
uv run python -I -B scripts/e4_pod_download.py > "$R/logs/download.log" 2>&1
grep -q ALLDONE "$R/logs/download.log" || { echo "download failed: see $R/logs/download.log"; exit 1; }
nohup uv run python -I -B "$R/scripts/e5_orchestrate.py" "$R" > "$R/logs/orchestrate.log" 2>&1 &
echo "orchestrator pid $!  (tail -f $R/logs/orchestrate.log)"
