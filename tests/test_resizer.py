import math

import pytest
import torch
import torch.nn.functional as F

from learnable_resizer.models import BilinearResizer, LearnableResizer, count_conv_weights

# Paper Table 2: number of parameters (in thousands) for filters n and residual blocks r.
PAPER_TABLE_2 = {
    (16, 1): 11.87, (16, 2): 16.48, (16, 3): 21.08, (16, 4): 25.69,
    (32, 1): 38.08, (32, 2): 56.51, (32, 3): 74.94, (32, 4): 93.37,
}  # fmt: skip


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize(
    ("in_size", "out_size"),
    [
        ((368, 368), (224, 224)),  # down-scaling (main paper setting)
        ((224, 224), (224, 224)),  # equal input/output
        ((56, 56), (224, 224)),  # 4x up-scaling (paper Table 9)
        ((480, 640), (192, 256)),  # non-square (paper Fig. 2)
        ((300, 200), (150, 150)),  # aspect ratio change
    ],
)
def test_output_shape(in_size, out_size):
    resizer = LearnableResizer(out_size)
    x = torch.randn(2, 3, *in_size)
    assert resizer(x).shape == (2, 3, *out_size)


def test_int_output_size_and_variable_input():
    resizer = LearnableResizer(150).eval()
    for h, w in [(300, 300), (213, 341), (100, 120)]:
        assert resizer(torch.randn(1, 3, h, w)).shape == (1, 3, 150, 150)


@pytest.mark.parametrize(("num_filters", "num_res_blocks"), sorted(PAPER_TABLE_2))
def test_conv_weight_count_matches_paper_table_2(num_filters, num_res_blocks):
    resizer = LearnableResizer(224, num_filters=num_filters, num_res_blocks=num_res_blocks)
    count = count_conv_weights(resizer)

    n = num_filters
    expected = 3 * n * 49 + n * n + num_res_blocks * 2 * 9 * n * n + 9 * n * n + n * 3 * 49
    assert count == expected
    # The paper truncates (not rounds) to two decimals, e.g. 21,088 -> 21.08.
    assert math.floor(count / 10) / 100 == PAPER_TABLE_2[(num_filters, num_res_blocks)]


@pytest.mark.parametrize("train", [True, False])
def test_zero_init_last_equals_bilinear(train):
    resizer = LearnableResizer((224, 224), zero_init_last=True).train(train)
    x = torch.randn(2, 3, 368, 368)
    expected = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    torch.testing.assert_close(resizer(x), expected)


def test_random_init_differs_from_bilinear():
    resizer = LearnableResizer((224, 224))
    x = torch.randn(2, 3, 368, 368)
    assert not torch.allclose(resizer(x), BilinearResizer((224, 224))(x))


def test_gradients_reach_all_parameters():
    resizer = LearnableResizer((64, 64), num_res_blocks=2)
    x = torch.randn(2, 3, 96, 96, requires_grad=True)
    resizer(x).square().mean().backward()
    for name, p in resizer.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert x.grad is not None


def test_bilinear_resizer():
    resizer = BilinearResizer((224, 224))
    assert sum(p.numel() for p in resizer.parameters()) == 0

    x = torch.randn(2, 3, 368, 368)
    expected = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    torch.testing.assert_close(resizer(x), expected)

    same = torch.randn(2, 3, 224, 224)
    assert resizer(same) is same


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic", "nearest", "area"])
def test_interpolation_modes(interpolation):
    resizer = LearnableResizer((32, 32), interpolation=interpolation)
    assert resizer(torch.randn(1, 3, 64, 64)).shape == (1, 3, 32, 32)


def test_antialias_requires_supported_mode():
    with pytest.raises(ValueError):
        BilinearResizer(32, interpolation="nearest", antialias=True)(torch.randn(1, 3, 64, 64))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_autocast_channels_last():
    resizer = LearnableResizer((224, 224)).cuda().to(memory_format=torch.channels_last)
    x = torch.randn(4, 3, 368, 368, device="cuda").to(memory_format=torch.channels_last)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = resizer(x)
    assert out.shape == (4, 3, 224, 224)
    out.float().mean().backward()
