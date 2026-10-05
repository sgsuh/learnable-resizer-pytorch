import pytest
import torch

from learnable_resizer.config import load_config
from learnable_resizer.data import build_transforms
from learnable_resizer.models import BilinearResizer, LearnableResizer, build_model
from learnable_resizer.utils import load_backbone_weights, save_checkpoint, topk_correct


@pytest.mark.parametrize("backbone", ["resnet50", "densenet121", "mobilenet_v2", "efficientnet_b0"])
@pytest.mark.parametrize("resizer_type", ["bilinear", "learnable"])
def test_build_model_forward(backbone, resizer_type):
    cfg = load_config(overrides=[f"model.backbone={backbone}", "model.pretrained=false",
                                 f"model.resizer.type={resizer_type}", "model.resizer.output_size=64"])
    model = build_model(cfg["model"]).eval()
    expected = LearnableResizer if resizer_type == "learnable" else BilinearResizer
    assert isinstance(model.resizer, expected)
    assert model(torch.randn(2, 3, 96, 96)).shape == (2, 10)


def test_load_backbone_weights(tmp_path):
    cfg = load_config(overrides=["model.backbone=mobilenet_v2", "model.pretrained=false"])
    source = build_model(cfg["model"])
    save_checkpoint(tmp_path / "ckpt.pt", model=source.state_dict())

    cfg = load_config(overrides=["model.backbone=mobilenet_v2", "model.pretrained=false",
                                 "model.resizer.type=learnable"])
    target = build_model(cfg["model"])
    load_backbone_weights(target, tmp_path / "ckpt.pt")
    for (name, a), b in zip(source.backbone.state_dict().items(), target.backbone.state_dict().values()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output(train):
    from PIL import Image

    cfg = load_config(overrides=["data.input_size=368"])
    out = build_transforms(cfg["data"], train=train)(Image.new("L", (500, 300)))
    assert out.shape == (3, 368, 368) and out.dtype == torch.float32


def test_topk_correct():
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.05]])
    assert topk_correct(logits, torch.tensor([1, 2]), ks=(1, 2)) == [1, 1]


def test_lr_scheduler_warmup_and_step_decay():
    from learnable_resizer.utils import build_lr_scheduler

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = build_lr_scheduler(optimizer, steps_per_epoch=10, step_epochs=2, gamma=0.5, warmup_epochs=1)
    lrs = []
    for _ in range(50):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.01)  # warmup step 1/10
    assert lrs[9] == pytest.approx(0.1)  # warmup done
    assert lrs[19] == pytest.approx(0.1)  # end of epoch 1
    assert lrs[20] == pytest.approx(0.05)  # decayed after 2 epochs
    assert lrs[40] == pytest.approx(0.025)
