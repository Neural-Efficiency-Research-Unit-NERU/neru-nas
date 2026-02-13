#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-cpu}"
SUPERNET_EPOCHS="${SUPERNET_EPOCHS:-1}"
ROUTER_EPOCHS="${ROUTER_EPOCHS:-1}"

export PYTHONPATH="$ROOT_DIR/src/models/router${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p artifacts

echo "[preflight] Checking required Python packages"
MISSING_PKGS="$("$PYTHON_BIN" -c '
import importlib.util
mods = ["torch", "torchvision", "onnx", "onnxruntime", "numpy", "tqdm", "matplotlib"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(" ".join(missing))
')"

if [ -n "$MISSING_PKGS" ]; then
  echo "Missing Python packages: $MISSING_PKGS"
  echo "Install them with:"
  echo "  $PYTHON_BIN -m pip install $MISSING_PKGS"
  exit 1
fi

echo "[1/5] Training Big/Little models"
"$PYTHON_BIN" src/models/supernet/train_supernet.py \
  --device "$DEVICE" \
  --epochs "$SUPERNET_EPOCHS"

echo "[2/5] Exporting ONNX models"
"$PYTHON_BIN" src/deployment/export_onnx.py

echo "[3/5] Measuring latency"
"$PYTHON_BIN" src/evaluation/measure_latency.py

echo "[4/5] Training router"
"$PYTHON_BIN" src/models/router/train_router.py \
  --device "$DEVICE" \
  --epochs "$ROUTER_EPOCHS"

echo "[5/5] Evaluating routing and generating plots"
"$PYTHON_BIN" src/evaluation/eval_routing.py

echo "Pipeline complete. Artifacts are in $ROOT_DIR/artifacts"
