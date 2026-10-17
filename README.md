# learnable-resizer-pytorch

A PyTorch implementation of the learnable image resizer from
**"Learning to Resize Images for Computer Vision Tasks"** (Hossein Talebi and Peyman
Milanfar, ICCV 2021, [arXiv:2103.09950](https://arxiv.org/abs/2103.09950)), with a
controlled reproduction study on Imagenette.

The resizer is a small CNN (about 12K parameters) that replaces the fixed bilinear
resize in front of a classifier. Its output is a bilinear resize of the input plus a
learned residual, and it is trained jointly with the backbone on the classification
loss only.

```
x ─┬─ bilinear resize ─────────────────────────────────────────────┐
   └─ conv7x7 → conv1x1 → BN → resize → res blocks → conv7x7 ─── + ─┴─→ backbone
```

## Findings

On Imagenette with a ResNet-50 trained from scratch, **the learnable resizer did not
improve accuracy** at any output resolution we tried.

| output resolution | bilinear (same input and skip path) | learnable resizer |
|---|---|---|
| 224 (input 368) | 87.34 ± 0.05 | 87.31 ± 0.18 |
| 160 (input 448) | 86.41 ± 0.16 | 85.93 ± 0.21 |
| 128 (input 368) | 85.17 ± 0.31 | 84.54 ± 0.08 |

Imagenette val top-1 (%), mean of the last 3 epochs, 3 seeds, BN statistics
recomputed before evaluation. At 224 the bilinear row is the control resized in the
data pipeline; this matches the skip-path control at 128 and 160 within 0.1 points.

The analysis found:

- **Val drops come from BatchNorm statistics.** Resizer runs show sporadic val drops
  caused by stale backbone BN running statistics. Recomputing them before each
  evaluation (`train.bn_recal_batches`) removes the drops.
- **Aliasing is not a factor.** Bilinear with and without antialiasing gives the same
  accuracy.
- **The residual's scale is the problem.** The unconstrained residual grows to ~4× the
  image scale and costs accuracy. Constraining it (`train.resizer_lr_scale`,
  `train.residual_l2`, `zero_init_last`) recovers 40–70% of the loss, but never beats
  bilinear.

See [docs/REPORT.md](docs/REPORT.md) for the full write-up, and
[docs/RESULTS.md](docs/RESULTS.md) for all tables.

## Features

- `LearnableResizer` (paper Fig. 3) for arbitrary input/output sizes, and a
  parameter-free `BilinearResizer` with the same interface. Both use TF2-compatible
  resizing (`align_corners=False`).
- Parameter counts are verified against the paper's Table 2 in the tests.
- Torchvision backbones (ResNet-50, DenseNet-121, MobileNet-v2, EfficientNet-B0, …)
  with a replaced classifier head.
- YAML configs with command-line overrides; unknown keys are rejected.
- bf16 AMP, `channels_last`, TensorBoard logging, including resizer output grids.
- Diagnostics for BatchNorm statistics and residual size (`scripts/diagnose.py`).

## Quick start

Everything runs inside Docker (NVIDIA GPU required).

```bash
# Build the image
docker compose -f docker/docker-compose.yml build dev

# Download Imagenette (full size) into ./data
docker run --rm -v "$PWD/data:/data" -u "$(id -u):$(id -g)" alpine:3 sh -c \
  "cd /data && wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz \
   && tar xzf imagenette2.tgz && rm imagenette2.tgz"

# Run the tests
docker compose -f docker/docker-compose.yml run --rm dev pytest -q

# Stage 1: bilinear ResNet-50 baseline from scratch (output 224)
docker compose -f docker/docker-compose.yml run --rm dev \
  python scripts/train.py --config configs/imagenette_resnet50_bilinear.yaml

# Stage 2: learnable resizer (368 -> 224) fine-tuned from the stage 1 checkpoint
docker compose -f docker/docker-compose.yml run --rm dev \
  python scripts/train.py --config configs/imagenette_resnet50_resizer.yaml \
  --opts train.bn_recal_batches=50

# TensorBoard at http://localhost:6006
docker compose -f docker/docker-compose.yml up tensorboard
```

Any config key can be overridden with `--opts key=value`, e.g.
`--opts data.input_size=448 model.resizer.output_size=160`. Outputs go to
`outputs/<run_name>/`. Scripts for every experiment in the report are listed in
[docs/REPORT.md § 6](docs/REPORT.md#6-reproducing).

Using the resizer on its own:

```python
import torch
from learnable_resizer.models import LearnableResizer

resizer = LearnableResizer(output_size=224, num_filters=16, num_res_blocks=1)
out = resizer(torch.randn(8, 3, 368, 368))  # -> (8, 3, 224, 224)
```

## Repository layout

```
configs/                 stage 1 / stage 2 YAML configs
docker/                  Dockerfile and compose services (dev, tensorboard)
docs/DESIGN.md           method summary, protocol, deviations from the paper
docs/RESULTS.md          all experiment tables
docs/REPORT.md           write-up of the reproduction study
scripts/train.py         training entry point
scripts/run_*.sh         experiment runners
scripts/diagnose.py      BN / residual diagnostics from per-epoch checkpoints
scripts/summarize.py     result tables across runs
src/learnable_resizer/   models, data, training engine, diagnostics
tests/                   unit tests
```

## Citation

```bibtex
@inproceedings{talebi2021learning,
  title     = {Learning to Resize Images for Computer Vision Tasks},
  author    = {Talebi, Hossein and Milanfar, Peyman},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021}
}
```

Reference implementations: the Keras code example
[Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)
and [sayakpaul/Learnable-Image-Resizing](https://github.com/sayakpaul/Learnable-Image-Resizing) (TF2).

## License

Apache License 2.0. See [LICENSE](LICENSE).
