#!/usr/bin/env bash
# All training runs and evaluations behind the README tables, in order.
# Each training run takes at most ~35 min on an Apple M1 (MPS).
# Usage: ./scripts/run_experiments.sh [run ...]   (default: all runs)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
COMMON=(--epochs 12 --samples-per-epoch 120000)
CNR_ADAPT_DAYS=2015-11-12,2015-11-16,2015-11-20   # first 3 CNR-EXT days (parking/splits.py)

run() {
  local name=$1; shift
  echo "=== $name ==="
  "$PY" train.py --name "$name" "$@"
  "$PY" evaluate.py --checkpoint "runs/$name/best.pt" --out "results/$name"
}

SELECTED="$*"
want() { [ -z "$SELECTED" ] || [[ " $SELECTED " == *" $1 "* ]]; }

# 1. main model: MobileNetV3-small (ImageNet), standard augmentation
want mobilenetv3 && run mobilenetv3 --arch mobilenet_v3_small "${COMMON[@]}"
# 2. baseline: mAlexNet from scratch
want malexnet && run malexnet --arch malexnet --lr 3e-3 "${COMMON[@]}"
# 3. weather shift: train and select on sunny frames only
want mobilenetv3_sunny && run mobilenetv3_sunny --arch mobilenet_v3_small \
  --train-filter weather=sunny --val-filter weather=sunny "${COMMON[@]}"
# 4. gap reduction (a): strong augmentation, still PKLot only
want mobilenetv3_strong && run mobilenetv3_strong --arch mobilenet_v3_small --augment strong "${COMMON[@]}"
# 5. gap reduction (b): brief fine-tuning of model 1 on the first 3 CNR-EXT days
want mobilenetv3_cnr_ft && run mobilenetv3_cnr_ft --init runs/mobilenetv3/best.pt --train-set cnr_ext \
  --train-filter "date=$CNR_ADAPT_DAYS" --val-set none --epochs 2 --lr 3e-4 --warmup-epochs 0
echo "ALL_EXPERIMENTS_DONE"
