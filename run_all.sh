#!/usr/bin/env bash
set -euo pipefail

# One-click pipeline:
# 1) Train conditional WGAN-GP
# 2) Generate scenarios for a target label
# 3) Plot load/wind/solar curves in one figure
# 4) Evaluate generation quality metrics and visualizations

DATA="shandong_gan_ready_final.csv"
EPOCHS=800
BATCH_SIZE=16
NOISE_DIM=128
N_CRITIC=5
GP_LAMBDA=10.0
LR=1e-4
SEED=42
SAVE_DIR="outputs"
SCALER_PATH="shandong_scaler.pkl"

LABEL=1
NUM_SAMPLES=20
GEN_OUTPUT="generated_samples.csv"
PLOT_OUTPUT="generated_curves.png"
SAMPLE_INDEX=0

SKIP_TRAIN=0
CHECKPOINT=""
ENABLE_EVAL=1
EVAL_OUT_DIR="evaluation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA="$2"; shift 2 ;;
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    --noise-dim)
      NOISE_DIM="$2"; shift 2 ;;
    --n-critic)
      N_CRITIC="$2"; shift 2 ;;
    --gp-lambda)
      GP_LAMBDA="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --save-dir)
      SAVE_DIR="$2"; shift 2 ;;
    --scaler-path)
      SCALER_PATH="$2"; shift 2 ;;
    --label)
      LABEL="$2"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --gen-output)
      GEN_OUTPUT="$2"; shift 2 ;;
    --plot-output)
      PLOT_OUTPUT="$2"; shift 2 ;;
    --sample-index)
      SAMPLE_INDEX="$2"; shift 2 ;;
    --skip-train)
      SKIP_TRAIN=1; shift 1 ;;
    --checkpoint)
      CHECKPOINT="$2"; shift 2 ;;
    --no-eval)
      ENABLE_EVAL=0; shift 1 ;;
    --eval-out-dir)
      EVAL_OUT_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: ./run_all.sh [options]"
      echo ""
      echo "Train options:"
      echo "  --data PATH"
      echo "  --epochs INT"
      echo "  --batch-size INT"
      echo "  --noise-dim INT"
      echo "  --n-critic INT"
      echo "  --gp-lambda FLOAT"
      echo "  --lr FLOAT"
      echo "  --seed INT"
      echo "  --save-dir DIR"
      echo "  --scaler-path PATH"
      echo ""
      echo "Generate/plot options:"
      echo "  --label {0|1|2}"
      echo "  --num-samples INT"
      echo "  --gen-output PATH"
      echo "  --plot-output PATH"
      echo "  --sample-index INT"
      echo ""
      echo "Pipeline control options:"
      echo "  --skip-train            Skip training and use checkpoint"
      echo "  --checkpoint PATH       Explicit checkpoint path (.pth)"
      echo "  --no-eval               Disable evaluation step"
      echo "  --eval-out-dir DIR      Evaluation output directory"
      exit 0 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

LATEST_RUN_DIR=""

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[1/4] Training conditional WGAN-GP..."
  python train.py \
    --data "$DATA" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --noise-dim "$NOISE_DIM" \
    --n-critic "$N_CRITIC" \
    --gp-lambda "$GP_LAMBDA" \
    --lr "$LR" \
    --seed "$SEED" \
    --save-dir "$SAVE_DIR" \
    --scaler-path "$SCALER_PATH"

  LATEST_RUN_DIR="$(ls -td "$SAVE_DIR"/* 2>/dev/null | head -n 1 || true)"
  if [[ -z "$LATEST_RUN_DIR" ]]; then
    echo "No run directory found under $SAVE_DIR"
    exit 1
  fi

  if [[ -z "$CHECKPOINT" ]]; then
    CHECKPOINT="$LATEST_RUN_DIR/cgan_wgangp_504.pth"
  fi
else
  echo "[1/4] Skipping training."
  if [[ -z "$CHECKPOINT" ]]; then
    LATEST_RUN_DIR="$(ls -td "$SAVE_DIR"/* 2>/dev/null | head -n 1 || true)"
    if [[ -z "$LATEST_RUN_DIR" ]]; then
      echo "No run directory found under $SAVE_DIR, and --checkpoint not provided"
      exit 1
    fi
    CHECKPOINT="$LATEST_RUN_DIR/cgan_wgangp_504.pth"
  fi
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT"
  exit 1
fi

echo "[2/4] Generating scenarios..."
python generate.py \
  --checkpoint "$CHECKPOINT" \
  --label "$LABEL" \
  --num-samples "$NUM_SAMPLES" \
  --noise-dim "$NOISE_DIM" \
  --scaler-path "$SCALER_PATH" \
  --output "$GEN_OUTPUT"

echo "[3/4] Plotting generated load/wind/solar curves..."
python plot_results.py \
  --input "$GEN_OUTPUT" \
  --sample-index "$SAMPLE_INDEX" \
  --output "$PLOT_OUTPUT"

if [[ "$ENABLE_EVAL" -eq 1 ]]; then
  echo "[4/4] Evaluating scenario-generation quality..."
  python evaluate.py \
    --real-data "$DATA" \
    --generated-data "$GEN_OUTPUT" \
    --label "$LABEL" \
    --out-dir "$EVAL_OUT_DIR"
else
  echo "[4/4] Evaluation skipped (--no-eval)."
fi

echo "Done."
if [[ -n "$LATEST_RUN_DIR" ]]; then
  echo "Run dir:    $LATEST_RUN_DIR"
fi
echo "Checkpoint: $CHECKPOINT"
echo "CSV:        $GEN_OUTPUT"
echo "Figure:     $PLOT_OUTPUT"
if [[ "$ENABLE_EVAL" -eq 1 ]]; then
  echo "Eval dir:   $EVAL_OUT_DIR"
fi
