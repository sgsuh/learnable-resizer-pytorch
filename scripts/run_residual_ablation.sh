#!/usr/bin/env bash
# Residual scale ablation for the learnable resizer at 368 -> 128 (run inside the dev container):
#   docker compose -f docker/docker-compose.yml run --rm dev bash scripts/run_residual_ablation.sh
# Each variant constrains the learned residual differently; compare against
# s2out128_skip368_* (same skip path, no residual) and s2out128_resizer368_* (unconstrained).
# Requires the stage 1 baseline from run_lowres.sh. Finished runs are skipped.
#   SEEDS     default "0 1 2"
#   VARIANTS  subset of variant tags to run (default: all, in the order below)
set -euo pipefail

SEEDS=(${SEEDS:-0 1 2})

declare -A VARIANT_OPTS=(
  [free]=""  # unconstrained, seed 0 only: residual RMS reference for s2out128_resizer368_*
  [lr0.1]="train.resizer_lr_scale=0.1"
  [lr0.01]="train.resizer_lr_scale=0.01"
  [l2_0.01]="train.residual_l2=0.01"
  [l2_0.1]="train.residual_l2=0.1"
  [zinit_lr0.01]="model.resizer.zero_init_last=true train.resizer_lr_scale=0.01"
)
VARIANTS=(${VARIANTS:-free lr0.1 lr0.01 l2_0.01 l2_0.1 zinit_lr0.01})

for variant in "${VARIANTS[@]}"; do
  seeds=("${SEEDS[@]}")
  [[ ${variant} == free ]] && seeds=(0)
  for seed in "${seeds[@]}"; do
    name="s2out128_res368_${variant}_seed${seed}"
    if [[ -f "outputs/${name}/summary.json" ]]; then
      echo "skip ${name} (done)"
      continue
    fi
    rm -rf "outputs/${name}"
    echo "=== ${name} ==="
    # shellcheck disable=SC2086  # VARIANT_OPTS holds space-separated overrides
    if python scripts/train.py --config configs/imagenette_resnet50_resizer.yaml --opts \
        run_name="${name}" seed="${seed}" \
        data.input_size=368 model.resizer.output_size=128 \
        model.init_checkpoint=outputs/imagenette_resnet50_bilinear128/last.pt \
        train.bn_recal_batches=50 ${VARIANT_OPTS[${variant}]} \
        > "outputs/${name}.log" 2>&1; then
      tail -n 1 "outputs/${name}.log"
    else
      echo "FAILED ${name}"; tail -n 5 "outputs/${name}.log"
    fi
  done
done

python scripts/summarize.py --pattern "s2out128_*"
