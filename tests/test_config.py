import pytest

from learnable_resizer.config import DEFAULTS, load_config


def test_defaults_without_file():
    cfg = load_config()
    assert cfg["train"]["lr"] == DEFAULTS["train"]["lr"]
    assert cfg["run_name"] == "default"


def test_yaml_and_overrides(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("data:\n  input_size: 368\nmodel:\n  resizer:\n    type: learnable\n")
    cfg = load_config(path, ["train.lr=0.01", "model.resizer.num_res_blocks=2", "model.init_checkpoint=null"])
    assert cfg["run_name"] == "exp"
    assert cfg["data"]["input_size"] == 368
    assert cfg["model"]["resizer"]["type"] == "learnable"
    assert cfg["train"]["lr"] == 0.01
    assert cfg["model"]["resizer"]["num_res_blocks"] == 2
    assert cfg["model"]["init_checkpoint"] is None
    # Defaults are not mutated.
    assert DEFAULTS["data"]["input_size"] == 224


@pytest.mark.parametrize("override", ["train.lrr=0.1", "model.resizer=1", "train.lr"])
def test_invalid_overrides(override):
    with pytest.raises((KeyError, TypeError, ValueError)):
        load_config(overrides=[override])
