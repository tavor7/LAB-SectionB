#!/usr/bin/env bash
# Wait for all DEFAULT_SWEEP_GRID variants to be complete, then eval + ship winner.
set -euo pipefail
cd "$(dirname "$0")/.."

EXPECTED=3

echo "Watching artifacts_sweep for ${EXPECTED} complete variants (400, 320, 240)..."
while true; do
  n=$(python -c "
from artifact_registry import DEFAULT_SWEEP_GRID, is_variant_complete, variant_dir
print(sum(1 for cw, ov in DEFAULT_SWEEP_GRID if is_variant_complete(variant_dir(cw, ov))))
")
  echo "$(date -Iseconds) complete=${n}/${EXPECTED}"
  if [[ "$n" -ge "$EXPECTED" ]]; then
    break
  fi
  sleep 300
done

echo "Running eval..."
python -u scripts/sweep_chunk_sizes.py eval --folds 5
echo "Shipping winner..."
python -u scripts/sweep_chunk_sizes.py ship
echo "Final public eval..."
python -u scripts/eval_public.py
