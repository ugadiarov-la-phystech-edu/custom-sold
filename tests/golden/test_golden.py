"""Compare outputs of the released checkpoints and of SOLD's losses with stored references (see references.py)."""

import pytest
import torch

from tests.conftest import require_checkpoint
from tests.golden import references

TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _assert_matches(name, actual):
    expected = torch.load(references.DATA_DIR / f"{name}.pt")
    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].shape == expected[key].shape, key
        assert torch.allclose(actual[key], expected[key], **TOLERANCE), \
            f"{name}/{key}: max abs diff {(actual[key] - expected[key]).abs().max().item():.3g}"


@pytest.mark.parametrize("name", sorted(references.SAVI_CHECKPOINTS))
def test_released_savi_checkpoint(name):
    path = require_checkpoint(references.SAVI_CHECKPOINTS[name])
    _assert_matches(name, references.savi_outputs(path))


def test_released_sold_checkpoint():
    path = require_checkpoint(references.SOLD_CHECKPOINT)
    _assert_matches("sold_reach_red", references.sold_outputs(path))


def test_released_sold_checkpoint_loads_every_weight():
    path = require_checkpoint(references.SOLD_CHECKPOINT)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    sold = references.load_sold(path)
    assert set(sold.state_dict()) == set(checkpoint["state_dict"])
    assert (sold.num_steps, sold.num_episodes) == (checkpoint["num_steps"], checkpoint["num_episodes"] - 1)


def test_tiny_sold_losses():
    _assert_matches("tiny_sold_losses", references.tiny_sold_losses())
