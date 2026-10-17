# Report: Reproducing "Learning to Resize Images for Computer Vision Tasks" on Imagenette

This report summarizes the experiments in this repository. Full tables, per-epoch
numbers and the exact protocol are in [`RESULTS.md`](RESULTS.md) and
[`DESIGN.md`](DESIGN.md).

## TL;DR

- We implemented the learnable resizer of Talebi & Milanfar (ICCV 2021) in PyTorch and
  tested it on Imagenette with a ResNet-50 trained from scratch.
- **The learnable resizer did not improve accuracy at any output resolution we tried
  (224, 160, 128).** At 224 it matches bilinear resizing; at 160 and 128 it is
  0.5–0.7 points below a bilinear model that uses exactly the same input and skip path.
- We traced the behaviour to three causes:
  1. **Sporadic val drops** in resizer runs come from stale backbone BatchNorm running
     statistics. Recomputing them before evaluation removes the drops, but not the gap.
  2. **Aliasing in the bilinear skip path is not a factor.** Bilinear with and without
     antialiasing, and resizing in the data pipeline, all give the same accuracy.
  3. **The unconstrained residual grows to ~4× the image scale** and costs accuracy.
     Constraining it (lower resizer LR, L2 penalty, zero init) recovers 40–70% of the loss,
     but no constrained variant beats plain bilinear.
- In this setting (10 easy classes, a backbone pre-trained on bilinear inputs, a
  10-epoch joint fine-tuning stage), resize quality does not seem to be the bottleneck.

## 1. Background

The paper replaces the fixed resizer in front of a classifier with a small CNN
(about 12K parameters). Its output is a plain bilinear resize of the input plus a
learned residual. The resizer is trained jointly with the backbone on the task loss
only, so it can learn "machine-friendly" images rather than visually pleasing ones.
The paper reports, for example, ResNet-50 on ImageNet improving from 24.7% to 23.0%
top-1 error with a 368 → 224 learnable resizer.

## 2. Setup

| item | choice |
|---|---|
| Dataset | Imagenette (full size), 10 ImageNet classes, 9,469 train / 3,925 val |
| Backbone | ResNet-50 (torchvision), trained **from scratch** |
| Resizer | `n = 16`, `r = 1`, random init (12,035 parameters) |
| Metric | val top-1, **mean of the last 3 epochs**, 3 seeds (mean ± std) |
| Hardware | RTX 4070 Laptop (8 GB), bf16 AMP, Docker |

**Why from scratch.** Imagenette is a subset of ImageNet train, so a pretrained
ImageNet backbone reaches 99.4% val accuracy after one epoch, which leaves no room to
measure differences.

**Two-stage protocol** (following the paper's fine-tuning recipe):

1. **Stage 1.** Train the backbone with plain bilinear resizing at the output
   resolution: 40 epochs, lr 0.05, 2-epoch warmup, ×0.94 every 2 epochs.
2. **Stage 2.** Start both arms from the same stage 1 checkpoint and train for 10
   epochs at lr 0.005:
   - **learnable:** randomly initialized resizer (e.g. 368 → 224) + backbone;
   - **control:** bilinear at the output resolution, same epochs and LR, so a gain
     cannot come from longer training.

## 3. Results

### 3.1 Output 224: no difference

| arm (3 seeds) | last-3 mean | final | best |
|---|---|---|---|
| bilinear 224 | 87.36 ± 0.23 | 87.08 ± 0.16 | 87.62 ± 0.19 |
| learnable 368 → 224 | 87.01 ± 0.88 | 87.58 ± 0.37 | 87.83 ± 0.05 |

The sign of the difference depends on the metric, and every difference is within noise.
A resizer input sweep (224–448, seed 0) showed no trend either.

### 3.2 Val drops come from BatchNorm running statistics

Only the resizer runs showed occasional val drops (to 82–85%, and once 72.6%) that
recovered in the next epoch while train accuracy stayed normal. Re-evaluating saved
checkpoints showed the following:

- Recomputing the **backbone** BN statistics on 50 train batches recovers most drops
  (e.g. 81.7 → 86.8). Recomputing only the resizer's BNs helps much less.
- The cause: BN running statistics are an EMA (momentum 0.1) and lag behind a resizer
  whose output distribution keeps changing during training.

We added `train.bn_recal_batches`, which recomputes all BN statistics before each
evaluation, and reran the 224 comparison with it in both arms:

| arm (3 seeds) | last-3 mean | worst epoch | worst epoch without recalibration |
|---|---|---|---|
| bilinear 224 | 87.34 ± 0.05 | 86.93 ± 0.13 | 86.66 ± 0.25 |
| learnable 368 → 224 | 87.31 ± 0.18 | 86.61 ± 0.23 | 77.27 ± 4.62 |

The drops disappear and the seed variance of the resizer arm shrinks from 0.88 to
0.18. The result is still a tie, so the drops were not hiding a gain. All later
experiments use recalibration.

### 3.3 Lower output resolution: the learnable resizer is worse

At 224 the task may be insensitive to resize quality, so we lowered the backbone
resolution. A new stage 1 baseline was trained at each output size.

| arm (3 seeds, last-3 mean) | output 128 (input 368) | output 160 (input 448) |
|---|---|---|
| bilinear, resized in the data pipeline | 85.26 ± 0.32 | 86.41 ± 0.30 |
| bilinear skip path, antialias on | 85.19 ± 0.36 | 86.25 ± 0.13 |
| bilinear skip path, antialias off | **85.17 ± 0.31** | **86.41 ± 0.16** |
| learnable (skip path + residual) | 84.54 ± 0.08 | 85.93 ± 0.21 |

- Lowering the resolution makes the task resolution-sensitive, as intended:
  bilinear goes 87.3 (224) → 86.4 (160) → 85.3 (128).
- The "skip path" rows use exactly the resizer's input and its plain bilinear resize,
  without the residual. All three bilinear variants agree within 0.16 points, so
  **aliasing in the 2.9× downscale does not matter**.
- Against its own skip path, **the learned residual costs 0.63 (128) and 0.48 (160)
  points**, about 3 standard errors each. Train accuracy is also slightly lower, so
  this is not overfitting.

### 3.4 Constraining the residual recovers the loss but gives no gain

In the 224 diagnostics the learned residual reached RMS 2–7, 2–6× the scale of the
image it is added to (plain resize output std 1.12). We added two options,
`train.resizer_lr_scale` and `train.residual_l2`, and compared constraints at 368 → 128:

| variant (368 → 128) | last-3 mean | residual RMS |
|---|---|---|
| bilinear skip path, no residual | **85.17 ± 0.31** | 0 |
| unconstrained | 84.54 ± 0.08 | 4.36 (seed 0) |
| resizer lr × 0.1 | 84.89 ± 0.18 | 0.97 |
| resizer lr × 0.01 | 84.85 ± 0.13 | 0.66 |
| residual L2 0.01 | 84.78 ± 0.36 | 0.46 |
| residual L2 0.1 | 84.99 ± 0.24 | 0.08 |
| zero-init + resizer lr × 0.01 | 84.91 ± 0.15 | 0.10 |

Every constraint shrinks the residual and recovers 0.24–0.45 of the 0.63-point loss
below the skip path (40–70%). Accuracy moves
toward the bilinear result as the residual shrinks toward zero. No variant exceeds it:
the best are 0.18–0.26 below, about one standard error.

## 4. Discussion

**What we can conclude.** On this setup the learnable resizer gives no benefit at
224, 160 or 128. An unconstrained residual is actively harmful at low resolution. The
harm comes from the residual's scale, not from BN artefacts or aliasing.

**Why the result may differ from the paper.** We did not test these explanations:

- **The dataset is easy and small.** Imagenette's 10 classes are visually distinct.
  The paper's gains are on ImageNet-1k, where fine detail separates many similar
  classes. Our backbone also overfits strongly (train ~97% vs val ~87%), so a better
  input may not translate into better generalization.
- **Short joint training.** Stage 2 is 10 epochs. The backbone, trained on bilinear
  inputs in stage 1, has little time to adapt to a substantially different input
  distribution. The paper fine-tunes on ImageNet, where each epoch is ~130× larger.
- **Nothing anchors the residual's scale.** The ResNet stem (conv → BN) normalizes its
  input, so a growing residual costs almost nothing in the loss. Constraints fix the
  scale but did not reveal a helpful residual.

**Deviations from the paper** (details in [`DESIGN.md`](DESIGN.md)): Imagenette
instead of ImageNet, backbones from scratch, a 2-epoch LR warmup in stage 1, standard
RandomResizedCrop + flip augmentation, and BN recalibration before evaluation (our
addition).

## 5. Possible next steps

- A harder fine-grained dataset (e.g. Imagewoof, 10 similar dog breeds).
- Longer stage 2 (e.g. 30 epochs) to test the adaptation-time explanation.
- Other backbones (DenseNet-121, MobileNet-v2) and resizer size (`n`, `r`) ablations.
- Visualizations of learned resizer outputs (paper Fig. 4).

## 6. Reproducing

All commands run in the Docker dev container (see the [README](../README.md)).
Runs that already have a `summary.json` are skipped, so each script can be re-run.

| experiment | command | run names |
|---|---|---|
| stage 1, output 224 | `python scripts/train.py --config configs/imagenette_resnet50_bilinear.yaml` | `imagenette_resnet50_bilinear` |
| 224 comparison + input sweep | `bash scripts/run_stage2.sh` | `s2_*` |
| 224 with BN recalibration | `PREFIX=s2bn SWEEP_SIZES= EXTRA_OPTS=train.bn_recal_batches=50 bash scripts/run_stage2.sh` | `s2bn_*` |
| val drop diagnosis | `python scripts/train.py ... --opts train.save_epoch_checkpoints=true`, then `python scripts/diagnose.py --run-dir outputs/<run>` | `diag_*` |
| stage 1 + 2 at 128 and 160 | `bash scripts/run_lowres.sh` | `imagenette_resnet50_bilinear{128,160}`, `s2out{128,160}_*` |
| same-skip-path controls | `bash scripts/run_skip_control.sh` | `s2out{128,160}_skip*` |
| residual scale ablation | `bash scripts/run_residual_ablation.sh` | `s2out128_res368_*` |
| summary tables | `python scripts/summarize.py --pattern "<prefix>_*"` | |
