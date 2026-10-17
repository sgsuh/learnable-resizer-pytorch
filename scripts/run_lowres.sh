#!/usr/bin/env bash
# Lower backbone resolution experiments (run inside the dev container):
#   docker compose -f docker/docker-compose.yml run --rm dev bash scripts/run_lowres.sh
# For each RESIZER_INPUT:OUT_SIZE pair, trains a stage 1 bilinear baseline from scratch
# at OUT_SIZE, then runs the stage 2 comparison (bilinear OUT_SIZE control vs learnable
# RESIZER_INPUT -> OUT_SIZE) with BN recalibration before every eval.
# Finished runs are skipped, so the script can be re-run.
#   PAIRS  space-separated input:output pairs (default "368:128 448:160")
set -euo pipefail

PAIRS=(${PAIRS:-368:128 448:160})

for pair in "${PAIRS[@]}"; do
  in_size=${pair%%:*}
  out_size=${pair##*:}
  stage1="imagenette_resnet50_bilinear${out_size}"
  if [[ -f "outputs/${stage1}/summary.json" ]]; then
    echo "skip ${stage1} (done)"
  else
    rm -rf "outputs/${stage1}"
    echo "=== ${stage1} ==="
    python scripts/train.py --config configs/imagenette_resnet50_bilinear.yaml \
      --opts run_name="${stage1}" data.input_size="${out_size}" model.resizer.output_size="${out_size}" \
      > "outputs/${stage1}.log" 2>&1
    tail -n 1 "outputs/${stage1}.log"
  fi

  PREFIX="s2out${out_size}" OUT_SIZE="${out_size}" RESIZER_INPUT="${in_size}" \
    STAGE1="outputs/${stage1}/last.pt" SWEEP_SIZES="" EXTRA_OPTS="train.bn_recal_batches=50" \
    bash scripts/run_stage2.sh
done
