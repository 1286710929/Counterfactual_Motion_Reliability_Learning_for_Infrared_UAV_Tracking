#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cmrtrack}"

TRACKER_NAME="ostrack"
TRACKER_PARAM="ostrack_motion_v166_single"
DATASET_NAME="antiuav410_test"
EPOCH=30
THREADS="${THREADS:-6}"
NUM_GPUS="${NUM_GPUS:-1}"

DEFAULT_PYTHON="/home/cyh/miniconda3/envs/focustrack5090/bin/python"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$DEFAULT_PYTHON" ]]; then
        PYTHON_BIN="$DEFAULT_PYTHON"
    else
        PYTHON_BIN="python"
    fi
fi

CKPT_SRC="$ROOT/checkpoints/OSTrack_ep0030.pth.tar"
CKPT_DST="$ROOT/output/checkpoints/train/ostrack/${TRACKER_PARAM}/OSTrack_ep0030.pth.tar"
CONFIG_SRC="$ROOT/configs/${TRACKER_PARAM}.yaml"
CONFIG_DST="$ROOT/experiments/ostrack/${TRACKER_PARAM}.yaml"
METRICS_LOG="$ROOT/eval/reproduced_antiuav410_test_metrics.log"

if [[ ! -f "$CKPT_SRC" ]]; then
    echo "[error] Missing checkpoint: $CKPT_SRC" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_DST" ]]; then
    mkdir -p "$(dirname "$CONFIG_DST")"
    cp -a "$CONFIG_SRC" "$CONFIG_DST"
fi

mkdir -p "$(dirname "$CKPT_DST")"
if [[ ! -f "$CKPT_DST" ]]; then
    ln -s "$CKPT_SRC" "$CKPT_DST" 2>/dev/null || cp -a "$CKPT_SRC" "$CKPT_DST"
fi

"$PYTHON_BIN" tracking/test.py \
    --tracker_name "$TRACKER_NAME" \
    --tracker_param "$TRACKER_PARAM" \
    --dataset_name "$DATASET_NAME" \
    --threads "$THREADS" \
    --num_gpus "$NUM_GPUS" \
    --run_id "$EPOCH" \
    --analysis 1 \
    --analysis_log "$METRICS_LOG" \
    --merge_results 1 \
    --skip_missing_seq 0
