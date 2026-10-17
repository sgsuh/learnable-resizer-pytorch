import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learnable_resizer.engine import Precision, ResidualTracker, train_one_epoch
from learnable_resizer.models import LearnableResizer
from learnable_resizer.models.classifier import ResizerClassifier
from learnable_resizer.utils import build_lr_scheduler, param_groups


def _tiny_model(zero_init_last=False):
    resizer = LearnableResizer(8, num_filters=4, zero_init_last=zero_init_last)
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
    return ResizerClassifier(resizer, backbone)


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass


def test_param_groups_resizer_lr_scale():
    model = _tiny_model()
    groups = param_groups(model, weight_decay=1e-4, lr=0.1, resizer_lr_scale=0.01)
    resizer_ids = {id(p) for p in model.resizer.parameters()}
    assert groups[0]["lr"] == 0.1 and not any(id(p) in resizer_ids for p in groups[0]["params"])
    for g in groups:
        in_resizer = {id(p) in resizer_ids for p in g["params"]}
        assert len(in_resizer) == 1
        assert g["lr"] == (0.1 * 0.01 if in_resizer.pop() else 0.1)
        assert g["weight_decay"] == (0.0 if g["params"][0].ndim <= 1 else 1e-4)
    covered = [id(p) for g in groups for p in g["params"]]
    assert sorted(covered) == sorted(id(p) for p in model.parameters())


def test_residual_tracker_matches_residual():
    resizer = LearnableResizer(8, num_filters=4).eval()
    x = torch.randn(2, 3, 16, 16)
    tracker = ResidualTracker(resizer)
    out = resizer(x)
    tracker.remove()
    expected = (out - resizer._resize(x)).pow(2).mean()
    torch.testing.assert_close(tracker.value, expected)
    assert not resizer._forward_hooks

    resizer = LearnableResizer(8, num_filters=4, zero_init_last=True)
    tracker = ResidualTracker(resizer)
    resizer(x)
    tracker.remove()
    assert tracker.value == 0


def test_train_one_epoch_residual_penalty_shrinks_residual():
    torch.manual_seed(1)
    images = torch.randn(32, 3, 16, 16)
    loader = DataLoader(TensorDataset(images, torch.randint(0, 10, (32,))), batch_size=8)
    device = torch.device("cpu")

    def run(residual_l2):
        torch.manual_seed(0)
        model = _tiny_model()
        optimizer = torch.optim.SGD(param_groups(model, 0.0, 0.05), lr=0.05)
        scheduler = build_lr_scheduler(optimizer, len(loader), step_epochs=2, gamma=1.0)
        for epoch in range(3):
            stats = train_one_epoch(model, loader, nn.CrossEntropyLoss(), optimizer, scheduler,
                                    Precision(False, device), device, epoch, _NullWriter(),
                                    channels_last=False, residual_l2=residual_l2)
        assert not model.resizer._forward_hooks
        tracker = ResidualTracker(model.resizer.eval())
        with torch.no_grad():
            model.resizer(images)
        tracker.remove()
        return stats, tracker.value.item()

    (plain_stats, plain_sq), (penalized_stats, penalized_sq) = run(0.0), run(10.0)
    assert "residual_sq" not in plain_stats and "residual_sq" in penalized_stats
    assert penalized_sq < 0.5 * plain_sq
