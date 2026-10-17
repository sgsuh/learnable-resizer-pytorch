#!/usr/bin/env bash
# Same-skip-path controls for the lower-resolution experiments (run inside the dev container):
#   docker compose -f docker/docker-compose.yml run --rm dev bash scripts/run_skip_control.sh
# The learnable resizer adds a learned residual to a plain bilinear resize of its input
# (e.g. 368 -> 128, antialias off). These runs keep that resize and drop the residual
# (model.resizer.type=bilinear on the same input), so comparing them with the resizer
# runs isolates the residual. A second set turns antialias on to measure aliasing.
# Requires the stage 1 baselines from run_lowres.sh. Finished runs are skipped.
#   PAIRS        space-separated input:output pairs (default "368:128 448:160")
#   SEEDS        default "0 1 2"
#   ANTIALIAS    antialias settings to run, in order (default "false true")
set -euo pipefail

PAIRS=(${PAIRS:-368:128 448:160})
SEEDS=(${SEEDS:-0 1 2})
ANTIALIAS=(${ANTIALIAS:-false true})

for antialias in "${ANTIALIAS[@]}"; do
  suffix=$([[ ${antialias} == true ]] && echo aa || echo "")
  for pair in "${PAIRS[@]}"; do
    in_size=${pair%%:*}
    out_size=${pair##*:}
    workers=$([[ ${in_size} -gt 368 ]] && echo 8 || echo 16)
    for seed in "${SEEDS[@]}"; do
      name="s2out${out_size}_skip${in_size}${suffix}_seed${seed}"
      if [[ -f "outputs/${name}/summary.json" ]]; then
        echo "skip ${name} (done)"
        continue
      fi
      rm -rf "outputs/${name}"
      echo "=== ${name} ==="
      if python scripts/train.py --config configs/imagenette_resnet50_bilinear_continued.yaml --opts \
          run_name="${name}" seed="${seed}" \
          data.input_size="${in_size}" data.num_workers="${workers}" \
          model.resizer.output_size="${out_size}" model.resizer.antialias="${antialias}" \
          model.init_checkpoint="outputs/imagenette_resnet50_bilinear${out_size}/last.pt" \
          train.bn_recal_batches=50 \
          > "outputs/${name}.log" 2>&1; then
        tail -n 1 "outputs/${name}.log"
      else
        echo "FAILED ${name}"; tail -n 5 "outputs/${name}.log"
      fi
    done
  done
done

for out_size in $(for pair in "${PAIRS[@]}"; do echo "${pair##*:}"; done); do
  python scripts/summarize.py --pattern "s2out${out_size}_*"
done
