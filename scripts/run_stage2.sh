#!/usr/bin/env bash
# Stage 2 experiments (run inside the dev container):
#   docker compose -f docker/docker-compose.yml run --rm dev bash scripts/run_stage2.sh
# Requires the stage 1 baseline at outputs/imagenette_resnet50_bilinear/last.pt.
# Finished runs (with summary.json) are skipped, so the script can be re-run.
# Environment variables:
#   PREFIX       run name prefix (default s2)
#   EXTRA_OPTS   config overrides added to every run, e.g. "train.bn_recal_batches=50"
#   SEEDS        seeds for the main comparison (default "0 1 2")
#   SWEEP_SIZES  resizer input sizes for the sweep; set to "" to skip it
set -euo pipefail

PREFIX=${PREFIX:-s2}
EXTRA_OPTS=(${EXTRA_OPTS:-})
SEEDS=(${SEEDS:-0 1 2})
SWEEP_SIZES=(${SWEEP_SIZES-224 256 320 448})

run() {
  local name=$1; shift
  if [[ -f "outputs/${name}/summary.json" ]]; then
    echo "skip ${name} (done)"
    return
  fi
  rm -rf "outputs/${name}"
  echo "=== ${name} ==="
  if python scripts/train.py "$@" --opts run_name="${name}" "${OPTS[@]}" "${EXTRA_OPTS[@]}" > "outputs/${name}.log" 2>&1; then
    tail -n 1 "outputs/${name}.log"
  else
    echo "FAILED ${name}"; tail -n 5 "outputs/${name}.log"
  fi
}

# Main comparison: bilinear control vs learnable resizer (368 -> 224), several seeds.
for seed in "${SEEDS[@]}"; do
  OPTS=(seed="${seed}")
  run "${PREFIX}_bilinear224_seed${seed}" --config configs/imagenette_resnet50_bilinear_continued.yaml
  run "${PREFIX}_resizer368_seed${seed}" --config configs/imagenette_resnet50_resizer.yaml
done

# Resizer input resolution sweep (output fixed at 224), seed 0.
for size in ${SWEEP_SIZES[@]+"${SWEEP_SIZES[@]}"}; do
  # Fewer workers: 16 workers x 2 prefetched 448px batches exceeded the 8 GB shm.
  OPTS=(seed=0 data.input_size="${size}" data.num_workers=8)
  run "${PREFIX}_resizer${size}_seed0" --config configs/imagenette_resnet50_resizer.yaml
done

python scripts/summarize.py --pattern "${PREFIX}_*"
