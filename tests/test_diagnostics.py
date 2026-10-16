import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learnable_resizer.diagnostics import batch_norms, bn_discrepancy, recalibrate_bn, resizer_output_stats
from learnable_resizer.models import LearnableResizer


def _loader(n=32, batch_size=8):
    images = torch.randn(n, 3, 8, 8) * 2 + 1
    return DataLoader(TensorDataset(images, torch.zeros(n, dtype=torch.long)), batch_size=batch_size)


def test_recalibrate_bn_matches_data_stats():
    model = nn.Sequential(nn.BatchNorm2d(3)).train()
    loader = _loader()
    recalibrate_bn(model, batch_norms(model).values(), loader, None, torch.device("cpu"), channels_last=False)
    bn = model[0]
    data = loader.dataset.tensors[0]
    torch.testing.assert_close(bn.running_mean, data.mean((0, 2, 3)), atol=1e-5, rtol=0)
    assert bn.momentum == 0.1 and model.training


def test_bn_discrepancy_zero_after_recalibration():
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4))
    loader = _loader()
    bns = batch_norms(model)
    recalibrate_bn(model, bns.values(), loader, None, torch.device("cpu"), channels_last=False)
    out = bn_discrepancy(model, bns, loader, None, torch.device("cpu"), channels_last=False)["1"]
    assert out["mean_shift"] < 1e-4
    # Recalibration averages per-batch (unbiased) variances, the discrepancy uses the pooled variance.
    assert 0.9 < out["var_ratio_min"] <= out["var_ratio_max"] < 1.1


def test_resizer_output_stats_zero_residual_at_init():
    resizer = LearnableResizer(4, zero_init_last=True)
    stats = resizer_output_stats(resizer, torch.randn(2, 3, 8, 8))
    assert stats["residual_rms"] == 0 and stats["eval_vs_batch_bn_rms"] == 0
    assert resizer.training
