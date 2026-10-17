#!/usr/bin/env bash
# Stage 2 experiments (run inside the dev container):
#   docker compose -f docker/docker-compose.yml run --rm dev bash scripts/run_stage2.sh
# Requires the stage 1 baseline (STAGE1, default outputs/imagenette_resnet50_bilinear/last.pt).
# Finished runs (with summary.json) are skipped, so the script can be re-run.
# Environment variables:
#   PREFIX         run name prefix (default s2)
#   EXTRA_OPTS     config overrides added to every run, e.g. "train.bn_recal_batches=50"
#   SEEDS          seeds for the main comparison (default "0 1 2")
#   SWEEP_SIZES    resizer input sizes for the sweep; set to "" to skip it
#   OUT_SIZE       backbone input resolution, i.e. resizer output (default 224)
#   RESIZER_INPUT  resizer input resolution in the main comparison (default 368)
#   STAGE1         stage 1 checkpoint both arms start from; must match OUT_SIZE
set -euo pipefail

PREFIX=${PREFIX:-s2}
EXTRA_OPTS=(${EXTRA_OPTS:-})
SEEDS=(${SEEDS:-0 1 2})
SWEEP_SIZES=(${SWEEP_SIZES-224 256 320 448})
OUT_SIZE=${OUT_SIZE:-224}
RESIZER_INPUT=${RESIZER_INPUT:-368}
STAGE1=${STAGE1:-outputs/imagenette_resnet50_bilinear/last.pt}
COMMON_OPTS=(model.resizer.output_size="${OUT_SIZE}" model.init_checkpoint="${STAGE1}")

run() {
  local name=$1; shift
  if [[ -f "outputs/${name}/summary.json" ]]; then
    echo "skip ${name} (done)"
    return
  fi
  rm -rf "outputs/${name}"
  echo "=== ${name} ==="
  if python scripts/train.py "$@" --opts run_name="${name}" "${COMMON_OPTS[@]}" "${OPTS[@]}" "${EXTRA_OPTS[@]}" \
      > "outputs/${name}.log" 2>&1; then
    tail -n 1 "outputs/${name}.log"
  else
    echo "FAILED ${name}"; tail -n 5 "outputs/${name}.log"
  fi
}

# Fewer workers above 368: 16 workers x 2 prefetched 448px batches exceeded the 8 GB shm.
workers_for() { [[ $1 -gt 368 ]] && echo 8 || echo 16; }

# Main comparison: bilinear control vs learnable resizer, several seeds.
for seed in "${SEEDS[@]}"; do
  OPTS=(seed="${seed}" data.input_size="${OUT_SIZE}")
  run "${PREFIX}_bilinear${OUT_SIZE}_seed${seed}" --config configs/imagenette_resnet50_bilinear_continued.yaml
  OPTS=(seed="${seed}" data.input_size="${RESIZER_INPUT}" data.num_workers="$(workers_for "${RESIZER_INPUT}")")
  run "${PREFIX}_resizer${RESIZER_INPUT}_seed${seed}" --config configs/imagenette_resnet50_resizer.yaml
done

# Resizer input resolution sweep (output fixed at OUT_SIZE), seed 0.
for size in ${SWEEP_SIZES[@]+"${SWEEP_SIZES[@]}"}; do
  OPTS=(seed=0 data.input_size="${size}" data.num_workers=8)
  run "${PREFIX}_resizer${size}_seed0" --config configs/imagenette_resnet50_resizer.yaml
done

python scripts/summarize.py --pattern "${PREFIX}_*"
