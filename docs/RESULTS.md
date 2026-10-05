# Results

All numbers are Imagenette val top-1 (%), 3,925 images. See `docs/DESIGN.md` for the
protocol.

## Stage 1: bilinear baseline from scratch

ResNet-50, bilinear 224 → 224, 40 epochs, lr 0.05 with 2-epoch warmup, batch 64.

| final top-1 | best top-1 | GFLOPs |
|---|---|---|
| 81.55 | 87.01 | 8.17 |

Val accuracy oscillated strongly near the end (84.5 → 87.0 → 84.4 → 81.6 over the last
four epochs). The LR is still about 0.019 at epoch 40.

## Stage 2: learnable resizer vs bilinear control

Both arms start from the stage 1 `last.pt`. Fine-tuning runs for 10 epochs at lr 0.005
with batch 64. The resizer is randomly initialized with `n=16, r=1` (12,035 params).

### Main comparison (3 seeds, mean ± std)

| config | final top-1 | last-3 mean top-1 | best top-1 | GFLOPs |
|---|---|---|---|---|
| bilinear 224 → 224 (control) | 87.08 ± 0.16 | 87.36 ± 0.23 | 87.62 ± 0.19 | 8.17 |
| learnable 368 → 224 | 87.58 ± 0.37 | 87.01 ± 0.88 | 87.83 ± 0.05 | 9.81 |

### Resizer input resolution sweep (seed 0, output 224)

| resizer input | final top-1 | last-3 mean top-1 | best top-1 | GFLOPs |
|---|---|---|---|---|
| 224 | 87.69 | 86.69 | 87.69 | 9.37 |
| 256 | 86.88 | 87.40 | 87.95 | 9.45 |
| 320 | 87.57 | 87.41 | 87.57 | 9.64 |
| 368 | 87.85 | 87.29 | 87.85 | 9.81 |
| 448 | 87.06 | 86.96 | 87.52 | 10.15 |

The 448 run used `data.num_workers=8`. With 16 workers it exhausted the container's
8 GB shared memory.

### Per-epoch val top-1

```
bilinear224 seed0  87.0 87.2 87.1 86.5 87.4 87.0 87.0 87.4 87.0 87.0
bilinear224 seed1  86.6 87.2 87.3 87.6 87.6 87.4 87.3 87.8 87.7 87.3
bilinear224 seed2  86.8 87.5 87.6 87.1 87.4 87.1 87.4 87.6 87.5 87.0
resizer368  seed0  86.3 87.0 87.8 84.8 87.3 87.0 87.1 87.0 87.0 87.8
resizer368  seed1  86.6 84.3 86.9 86.5 87.5 87.8 82.4 83.5 87.4 87.2
resizer368  seed2  86.4 85.2 86.2 87.4 87.1 87.0 86.9 87.9 87.5 87.7
resizer224  seed0  84.1 86.5 87.4 83.3 87.3 87.1 86.6 86.9 85.5 87.7
resizer448  seed0  86.7 85.7 72.6 86.7 87.0 87.2 87.4 86.3 87.5 87.1
```

## Observations

1. **No clear gain from the learnable resizer in this setting.** The sign of the
   difference depends on the metric:
   - final epoch: +0.5 pp
   - best epoch: +0.2 pp
   - last-3 mean: −0.35 pp

   All of these are within run-to-run noise. The paper reports +1.7 pp for ResNet-50
   (368 → 224) on ImageNet.
2. **Learnable-resizer runs have sporadic val drops** that the bilinear runs never show
   (82–85% in several runs, 72.6% for 448). They recover in the next epoch. Train
   accuracy shows no matching drop, so this is likely an evaluation-mode effect. Two
   candidates are the resizer's BatchNorm running statistics and large swings in the
   resizer output scale. Not yet diagnosed.
3. **No resolution trend.** The paper finds input > output generally helps. Here
   224–448 inputs are indistinguishable given the noise.
4. **The backbone is heavily overfit** (train ~97% vs val ~87%). The task may also be
   too coarse (10 easy classes at 224 px) for resize quality to matter.
