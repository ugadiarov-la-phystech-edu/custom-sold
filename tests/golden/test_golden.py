"""Compare outputs of the released checkpoints and of SOLD's losses with stored references (see references.py)."""

import pytest
from tests.conftest import make_online_trainer, require_checkpoint
from tests.golden import references
import torch

TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _assert_matches(name, actual):
    expected = torch.load(references.DATA_DIR / f"{name}.pt")
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert torch.allclose(actual[key], expected[key], **TOLERANCE), (
            f"{name}/{key}: max abs diff {(actual[key] - expected[key]).abs().max().item():.3g}"
        )


@pytest.mark.parametrize("name", sorted(references.SAVI_CHECKPOINTS))
def test_released_savi_checkpoint(name):
    path = require_checkpoint(references.SAVI_CHECKPOINTS[name])
    _assert_matches(name, references.savi_outputs(path))


def test_released_sold_checkpoint():
    path = require_checkpoint(references.SOLD_CHECKPOINT)
    _assert_matches("sold_reach_red", references.sold_outputs(path))


@pytest.mark.parametrize("name", ["released", "legacy"])
def test_sold_checkpoint_loads_every_weight(name):
    if name == "released":
        path = require_checkpoint(references.SOLD_CHECKPOINT)
        sold = references.load_sold(path)
    else:
        path = references.LEGACY_CHECKPOINT
        sold = references.load_legacy_sold()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert set(sold.state_dict()) == set(checkpoint["state_dict"])
    assert (sold.num_steps, sold.num_episodes) == (checkpoint["num_steps"], checkpoint["num_episodes"] - 1)


# Checkpoints from before the refactoring must keep loading, with the same outputs, and resume.
def test_legacy_sold_checkpoint():
    _assert_matches("legacy_tiny_sold", references.legacy_sold_outputs())


def test_legacy_sold_checkpoint_resumes_training(tmp_path):
    num_steps = torch.load(references.LEGACY_CHECKPOINT, map_location="cpu", weights_only=False)["num_steps"]
    sold = references.load_legacy_sold(max_steps=num_steps + 10)
    assert sold.finetune_autoencoder
    trainer = make_online_trainer(tmp_path)
    trainer.fit(sold, ckpt_path=references.LEGACY_CHECKPOINT)
    assert sold.num_steps == num_steps + 10
    assert len(trainer.optimizers) == 5  # Fine-tuning adds the autoencoder's optimizer, restored from the checkpoint.


def test_tiny_sold_losses():
    _assert_matches("tiny_sold_losses", references.tiny_sold_losses())


def test_tiny_sold_training():
    _assert_matches("tiny_sold_training", references.tiny_sold_training())
