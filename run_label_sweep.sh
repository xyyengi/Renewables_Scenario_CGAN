#!/usr/bin/env bash
set -euo pipefail

# Run scenario generation + evaluation for labels 0/1/2 without overwriting outputs.
# Default behavior: skip training and reuse a checkpoint.

DATA="shandong_gan_ready_final.csv"
CHECKPOINT=""
NUM_SAMPLES=100
NOISE_DIM=128
BASE_DIR="experiments"
TAG="run_$(date +%Y%m%d_%H%M%S)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA="$2"; shift 2 ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --noise-dim)
      NOISE_DIM="$2"; shift 2 ;;
    --base-dir)
      BASE_DIR="$2"; shift 2 ;;
    --tag)
      TAG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./run_label_sweep.sh [options]"
      echo "  --data PATH"
      echo "  --checkpoint PATH"
      echo "  --num-samples INT (default: 100)"
      echo "  --noise-dim INT (default: 128)"
      echo "  --base-dir DIR (default: experiments)"
      echo "  --tag NAME (default: timestamp)"
      exit 0 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  latest_run="$(ls -td outputs/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$latest_run" ]]; then
    echo "No checkpoint found in outputs/. Please pass --checkpoint PATH"
    exit 1
  fi
  CHECKPOINT="$latest_run/cgan_wgangp_504.pth"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

RUN_DIR="$BASE_DIR/$TAG"
mkdir -p "$RUN_DIR"

echo "Using checkpoint: $CHECKPOINT"
echo "Saving all outputs to: $RUN_DIR"

for label in 0 1 2; do
  echo "=== Label $label ==="
  ./run_all.sh \
    --skip-train \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --label "$label" \
    --num-samples "$NUM_SAMPLES" \
    --noise-dim "$NOISE_DIM" \
    --gen-output "$RUN_DIR/generated_label${label}.csv" \
    --plot-output "$RUN_DIR/generated_label${label}.png" \
    --eval-out-dir "$RUN_DIR/eval_label${label}"
done

python summarize_metrics.py --root "$RUN_DIR" --output "$RUN_DIR/metrics_summary.csv"

echo "Sweep completed: $RUN_DIR"
