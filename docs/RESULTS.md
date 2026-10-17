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
   (82–85% in several runs, 72.6% for 448). They recover in the next epoch, and train
   accuracy shows no matching drop. The cause is stale backbone BatchNorm running
   statistics; see "Diagnosis: eval-mode val drops" below.
3. **No resolution trend.** The paper finds input > output generally helps. Here
   224–448 inputs are indistinguishable given the noise.
4. **The backbone is heavily overfit** (train ~97% vs val ~87%). The task may also be
   too coarse (10 easy classes at 224 px) for resize quality to matter.

## Diagnosis: eval-mode val drops

Five learnable-resizer runs were repeated with `train.save_epoch_checkpoints=true`, and
every epoch checkpoint was analyzed with `scripts/diagnose.py`. For each checkpoint, val
top-1 is measured with the stored BN running statistics (`eval`) and after recomputing
them on 50 train batches (3,200 images). The recomputation covers the resizer BNs only,
the backbone BNs only, or both. The script also records how large the learned residual
is: resizer output minus plain bilinear resize, on 64 val images.

One standard error at 87% on 3,925 images is about 0.5 pp. Only differences above
about 1 pp are meaningful. Exact `eval` numbers can differ by about 0.05 pp from the
training log because bf16 inference is not deterministic.

### Per-epoch val top-1: stored vs recomputed backbone BN statistics

```
368 seed1          eval   81.7 85.3 87.6 87.2 87.3 86.8 85.4 87.1 87.4 87.4
                   recal  86.8 86.5 87.5 87.6 87.5 87.5 86.7 87.6 87.6 87.0
448 seed0          eval   83.4 87.4 86.9 86.3 87.4 88.1 87.2 87.1 87.6 87.5
                   recal  85.5 87.2 88.1 87.1 87.1 88.1 87.9 87.6 87.4 87.5
368 zero-init s0   eval   86.4 86.4 87.4 86.3 86.9 87.2 87.2 85.5 87.7 87.5
                   recal  87.3 86.6 87.8 87.2 87.0 87.4 87.3 86.7 87.4 87.5
368 zero-init s1   eval   82.0 80.9 86.4 87.4 86.9 86.1 88.0 87.3 85.9 87.5
                   recal  86.3 86.1 86.7 87.7 87.1 86.9 87.2 87.1 87.4 87.8
368 zero-init s2   eval   86.2 87.0 86.2 87.2 86.8 87.3 87.1 85.2 87.2 86.0
                   recal  86.8 87.1 87.1 87.9 87.5 87.5 87.5 87.3 87.1 87.5
```

### Summary per run

| run | min top-1 (eval / recal backbone / recal resizer / recal all) | last-3 mean (eval / recal all) | residual RMS |
|---|---|---|---|
| 368 seed1 | 81.68 / 86.55 / 83.08 / 86.65 | 87.33 / 87.45 | 2.73–3.42 |
| 448 seed0 | 83.41 / 85.45 / 84.00 / 85.22 | 87.40 / 87.55 | 2.20–3.34 |
| 368 zero-init seed0 | 85.53 / 86.57 / 84.92 / 86.37 | 86.90 / 87.23 | 2.93–3.91 |
| 368 zero-init seed1 | 80.87 / 86.14 / 84.89 / 86.68 | 86.86 / 87.37 | 6.00–6.88 |
| 368 zero-init seed2 | 85.25 / 86.78 / 86.14 / 86.68 | 86.14 / 87.37 | 1.93–2.40 |

For reference, the plain bilinear resize output has std 1.12 (normalized pixels).

### Findings

1. **Recomputing the backbone BN statistics removes most drops.** Examples:
   - 368 seed1, epoch 0: 81.7 → 86.8
   - zero-init seed1, epoch 1: 80.9 → 86.1
   - zero-init seed2, epoch 7: 85.2 → 87.3

   Recomputing only the resizer BNs helps much less. The one exception is 368 seed1,
   epoch 6, where both recomputations recover the drop. The drops come from the
   running statistics (EMA, momentum 0.1) lagging behind a resizer whose output
   distribution is still moving.
2. **The learned residual is large.** Its RMS is 2–7, i.e. 2–6× the scale of the
   bilinear image it is added to. So the backbone sees a re-scaled, substantially
   different input, not "bilinear plus a small correction". Small changes in the
   resizer, or in its BN statistics, shift the backbone input distribution
   noticeably. Switching the resizer's BNs from running to batch statistics changes
   its output by RMS 0.4–1.7.
3. **`zero_init_last=true` does not help.** The residual reaches RMS 1.9–6.1 within
   the first epoch, and drops still occur. The run with the largest residual
   (seed1) has the deepest drops.
4. **Likely root cause (not tested):** nothing constrains the residual scale. The
   ResNet stem normalizes its input scale (conv → BN), so a growing residual costs
   almost no loss.
5. **The drops do not explain the missing gain.** With all BN statistics recomputed,
   the last-3 mean is 87.2–87.6 for every run. That matches the bilinear control
   (87.36 ± 0.23).

## Stage 2 with BN recalibration before every eval

Same protocol as the main comparison, plus `train.bn_recal_batches=50`: before each
eval, the running statistics of every BN layer (resizer and backbone) are recomputed
on 50 train batches. Applied to both arms (for the bilinear control only the backbone
has BNs). Run names `s2bn_*`; `val_raw_*` in `metrics.jsonl` is the eval with the
stored (EMA) statistics.

`s2bn_resizer368_seed2` first started without a visible GPU (WSL driver glitch) and
fell back to CPU fp32; it was stopped and rerun on GPU. `train.py` now refuses to run
without CUDA unless `--allow-cpu` is given.

### Main comparison (3 seeds, mean ± std)

| metric | bilinear 224 | learnable 368→224 |
|---|---|---|
| final epoch | 87.19 ± 0.13 | 87.36 ± 0.29 |
| last-3 mean | 87.34 ± 0.05 | 87.31 ± 0.18 |
| best | 87.83 ± 0.19 | 87.76 ± 0.19 |
| worst epoch | 86.93 ± 0.13 | 86.61 ± 0.23 |
| last-3 mean, stored BN stats (`val_raw`) | 87.42 ± 0.13 | 86.75 ± 0.45 |
| worst epoch, stored BN stats (`val_raw`) | 86.66 ± 0.25 | 77.27 ± 4.62 |

### Findings

1. **Recalibration removes the drops.** The worst recalibrated epoch of every resizer
   run is ≥ 86.3, while the same runs evaluated with stored statistics still drop to
   73.7–82.5. The seed variance of the resizer arm shrinks from 0.88 to 0.18 (last-3
   mean, compared with the `s2_*` runs).
2. **It does not change the bilinear control.** Recalibrated and stored-statistics
   evals differ by at most ±0.8 per epoch, and the last-3 means agree (87.34 vs 87.42).
3. **There is still no gain from the learnable resizer at 224 output.** The last-3
   means are 87.31 ± 0.18 vs 87.34 ± 0.05, well within noise. The conclusion of the
   main comparison holds with the drops removed.
