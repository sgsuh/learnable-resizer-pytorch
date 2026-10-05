# Design: Learnable Image Resizer in PyTorch

PyTorch implementation of **"Learning to Resize Images for Computer Vision Tasks"**
(Talebi & Milanfar, ICCV 2021, [arXiv:2103.09950](https://arxiv.org/abs/2103.09950)).

References used for this design:

- Paper: https://arxiv.org/pdf/2103.09950
- Keras Code Example: https://keras.io/examples/vision/learnable_resizer/
- TF2 reference implementation: https://github.com/sayakpaul/Learnable-Image-Resizing

---

## 1. Summary of the method

A small CNN replaces the fixed (bilinear/bicubic) resizer in front of a vision
backbone. The resizer and the backbone are trained **jointly** using only the task loss.
No pixel or perceptual loss is applied to the resized image, so the resizer learns
"machine-friendly" images (typically over-sharpened high-frequency details) rather
than visually pleasing ones.

### 1.1 Architecture (paper Fig. 3)

Default configuration: `n = 16` filters, `r = 1` residual block, LeakyReLU slope `0.2`.

```
x ─┬─ Resize(x) ───────────────────────────────────────────────────────┐  (image skip)
   │                                                                   │
   └─ Conv7x7(n) → LReLU → Conv1x1(n) → LReLU → BN                      │
      → Resize (feature bottleneck) = b ──┬─ ResBlock × r               │
                                          │  → Conv3x3(n) → BN          │
                                          └──────── + ─────────         │  (feature skip)
      → Conv7x7(3) ─────────────────────────────────── + ◄─────────────┘
                                                       = output

ResBlock(z): Conv3x3(n) → BN → LReLU → Conv3x3(n) → BN → (+ z)
```

- All convolutions use stride 1 and "same" padding.
- Bias: following the Keras example, convs followed by BN have no bias. The first
  Conv7x7, the Conv1x1, and the last Conv7x7 have bias.
- `Resize` is bilinear by default. The paper reports results are barely sensitive to
  this choice (bicubic/Lanczos also work).
- The feature resize lets the model handle **arbitrary** output sizes and aspect
  ratios, for both up-scaling and down-scaling.

### 1.2 Parameter count (paper Table 2)

Table 2 counts **conv kernel weights only**. It excludes bias and BN parameters.
For example, `n=32, r=1` → 4704 + 1024 + 9216 + 4704 + 18432 = 38,080 = "38.08K".

| filters \ blocks | r=1   | r=2   | r=3   | r=4   |
|------------------|-------|-------|-------|-------|
| n=16             | 11.87K| 16.48K| 21.08K| 25.69K|
| n=32             | 38.08K| 56.51K| 74.94K| 93.37K|

A unit test checks that our conv-weight count reproduces this table.

### 1.3 Training recipe (paper)

| Item | Paper |
|---|---|
| Loss | Softmax cross-entropy with label smoothing `ε = 0.1` |
| Optimizer | SGD, momentum 0.9 |
| LR | 0.05 from scratch, **0.005 for fine-tuning** |
| LR schedule | ×0.94 every 2 epochs |
| Backbone init | Pre-trained baseline (trained with bilinear resizing) |
| Resizer init | Random |
| Data | Images are first resized (bilinear) to a fixed resizer-input resolution so they can be batched. Train and test use the same configuration. |
| Resolutions | Resizer input ∈ {224, 256, 320, 368, 448} → output 224 |
| Best setting | Usually input > output, e.g. 368 → 224 (ResNet-50: 24.7% → 23.0% top-1 err) |

### 1.4 Keras example / TF2 repo

- Cats vs Dogs, DenseNet-121 trained from scratch, input 300×300 → 150×150.
- Plain SGD (Keras default lr 0.01), label smoothing 0.1, batch size 64.
- Reported: bilinear 60.19% vs learnable 67.67% top-1 (TF2 repo, 10 epochs).
- Inputs are scaled to `[0, 1]` before the resizer.

---

## 2. Scope of this repository

| Decision | Choice |
|---|---|
| Dataset | **Imagenette** (10-class ImageNet subset, full-size `imagenette2`) |
| Config | argparse + YAML (CLI flags override YAML values) |
| Logging | TensorBoard |
| Environment | Docker only (all installs and runs happen inside the container) |
| Hardware target | Single RTX 4070 Laptop GPU (8 GB) |

Full ImageNet training is out of scope for this hardware. Imagenette keeps the paper's
setting (ImageNet-pretrained backbones fine-tuned with a new resizer) at a feasible
cost.

### Imagenette notes

- 10 classes, about 9.5K train / 3.9K val images. Folder names are ImageNet WNIDs.
- We use the full-size release (`imagenette2.tgz`) rather than `imagenette2-320`.
  With the 320 release (shorter side 320), a 368 or 448 resizer input would require
  up-sampling.
- Imagenette images are part of the ImageNet train set, so ImageNet-pretrained
  backbones have seen them. Absolute accuracy will be high. We care about the
  **relative** difference between bilinear and learnable resizers under identical
  conditions.

---

## 3. Repository layout

```
learnable-resizer-pytorch/
├── docker/
│   ├── Dockerfile             # pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime
│   └── docker-compose.yml     # dev (GPU) + tensorboard services
├── requirements.txt
├── configs/
│   ├── imagenette_resnet50_bilinear.yaml
│   ├── imagenette_resnet50_resizer.yaml
│   └── ...
├── src/learnable_resizer/
│   ├── models/
│   │   ├── resizer.py         # LearnableResizer, BilinearResizer, ResBlock
│   │   ├── backbones.py       # torchvision backbone factory (head replacement)
│   │   └── classifier.py      # ResizerClassifier = resizer + backbone
│   ├── data/
│   │   └── imagenette.py      # ImageFolder-based datasets + transforms
│   ├── config.py              # YAML loading + argparse overrides
│   ├── engine.py              # train / evaluate loops (AMP, channels_last)
│   ├── visualize.py           # bilinear vs learned vs |diff| figures (paper Fig. 4/6)
│   └── utils.py               # seeding, checkpointing, meters
├── scripts/
│   ├── train.py
│   └── visualize.py
├── tests/
│   └── test_resizer.py
├── data/                      # (gitignored) datasets
└── outputs/                   # (gitignored) checkpoints + TensorBoard logs
```

---

## 4. Module design

### 4.1 `LearnableResizer`

```python
class LearnableResizer(nn.Module):
    def __init__(
        self,
        output_size: tuple[int, int],
        in_channels: int = 3,
        num_filters: int = 16,
        num_res_blocks: int = 1,
        interpolation: str = "bilinear",
        antialias: bool = False,
        negative_slope: float = 0.2,
        zero_init_last: bool = False,
    ): ...

    def forward(self, x: Tensor) -> Tensor:  # (B, C, H, W) -> (B, C, *output_size)
```

- **Interpolation**: `F.interpolate(mode=..., align_corners=False, antialias=antialias)`.
  `align_corners=False` matches TF2's half-pixel-centers resizing.
  `antialias=False` by default to match TF/Keras `Resizing` and the paper's bilinear
  baseline. It is exposed as an option.
- **`zero_init_last`**: zero-initializes the final Conv7x7 so that the resizer is
  exactly the bilinear resizer at step 0. This is off by default because the paper
  uses random init. It is kept as an ablation for stability when fine-tuning a
  pre-trained backbone.

### 4.2 `BilinearResizer`

Baseline with the same interface and no parameters. Every experiment pair differs
**only** in the resizer module, so the comparison is controlled.

### 4.3 `ResizerClassifier`

```python
class ResizerClassifier(nn.Module):
    def __init__(self, resizer: nn.Module, backbone: nn.Module): ...
    def forward(self, x):            # x: normalized, at resizer-input resolution
        return self.backbone(self.resizer(x))
```

**Normalization happens before the resizer** (ImageNet mean/std, applied in the data
transform). Because of the image skip connection, the resizer output starts as
"bilinear-resized normalized image + learned residual". The pre-trained backbone
therefore receives the input distribution it expects. This differs from the Keras
example, which feeds `[0, 1]` images and trains the backbone from scratch.

### 4.4 Backbones

torchvision models with ImageNet weights and a replaced classification head
(10 classes):

| Paper model | Ours |
|---|---|
| ResNet-50 | `resnet50` (primary) |
| DenseNet-121 | `densenet121` |
| MobileNet-v2 | `mobilenet_v2` |
| EfficientNet-b0 (IQA) | `efficientnet_b0` |
| Inception-v2 | not available in torchvision; skipped |

---

## 5. Data pipeline

Following the paper, images are brought to a fixed **resizer-input resolution** `S_in`
in the data pipeline. The model then maps `S_in → S_out` (bilinear or learned).

| Split | Transform |
|---|---|
| train | `RandomResizedCrop(S_in, scale=(0.35, 1.0))` → `RandomHorizontalFlip` → `ToTensor` → `Normalize` |
| val   | `Resize(round(S_in / 0.875))` → `CenterCrop(S_in)` → `ToTensor` → `Normalize` |

The baseline uses `S_in = S_out = 224` with `BilinearResizer`, which is an identity
resize. That is equivalent to the standard torchvision pipeline. Each transform
choice is configurable in YAML.

---

## 6. Training

| Item | Setting |
|---|---|
| Loss | `nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 (BN and bias excluded) |
| LR | 0.005 (fine-tune) |
| Schedule | `StepLR(step_size=2, gamma=0.94)` (paper); configurable |
| Precision | AMP (bf16 if supported, else fp16 + GradScaler), `channels_last` |
| Batch size | Fixed across resolutions where memory allows (paper reduced it with resolution; recorded as a deviation) |
| Metrics | Top-1 / Top-5 accuracy, loss, resizer + backbone params, GFLOPs |
| Logging | TensorBoard scalars + periodic image grids of resizer outputs |
| Checkpoints | `outputs/<run_name>/{last,best}.pt` + resolved config YAML |

Paper protocol on Imagenette:

1. **Baseline**: fine-tune ImageNet-pretrained ResNet-50 with `BilinearResizer`, 224 → 224.
2. **Proposed**: initialize the backbone from step 1's checkpoint and the resizer
   randomly. Jointly train `S_in → 224` for `S_in ∈ {224, 256, 320, 368, 448}`.
3. **Fair control**: continue training the baseline for the same number of extra
   epochs as step 2. Any gain then cannot be attributed to longer training.

---

## 7. Experiment plan

| Phase | Goal | Done when |
|---|---|---|
| 1 | Docker environment + Imagenette download | GPU visible in container, dataset under `data/imagenette2` |
| 2 | `LearnableResizer` + tests | Shapes for up/down-scaling, Table 2 param counts reproduced |
| 3 | Training pipeline (config, data, engine, TensorBoard) | Baseline ResNet-50 run converges on Imagenette |
| 4 | Main comparison | Bilinear vs learned for 368 → 224 (+ resolution sweep) |
| 5 | Analysis | Fig. 4/6-style visualizations, `r`/`n` ablation, `zero_init_last` ablation, other backbones |
| 6 (optional) | IQA on AVA with EMD loss (`d = 2`) | — |

---

## 8. Running (Docker only)

```bash
# Build image
docker compose -f docker/docker-compose.yml build dev

# Download Imagenette (full size) into ./data
docker run --rm -v "$PWD/data:/data" -u "$(id -u):$(id -g)" alpine:3 sh -c \
  "cd /data && wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz \
   && tar xzf imagenette2.tgz && rm imagenette2.tgz"

# Tests
docker compose -f docker/docker-compose.yml run --rm dev pytest -q

# Train
docker compose -f docker/docker-compose.yml run --rm dev \
  python scripts/train.py --config configs/imagenette_resnet50_resizer.yaml

# TensorBoard at http://localhost:6006
docker compose -f docker/docker-compose.yml up tensorboard
```

---

## 9. Known deviations from the paper

- Imagenette (10 classes) instead of ImageNet-1k. Backbones are ImageNet-pretrained.
- No Inception-v2 backbone (not in torchvision).
- Batch size is not reduced with resolution unless memory requires it.
- Augmentation is not specified in the paper. We use standard RandomResizedCrop + flip.
- The paper mentions a "Sigmoid" on logits, but its Eq. (1) is a softmax cross-entropy
  with smoothed labels. We use softmax cross-entropy.
