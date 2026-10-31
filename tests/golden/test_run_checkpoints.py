"""Checkpoints of real training runs keep loading, with the same outputs, and resume (opt-in, see references.py).

SOLD_RUN_CHECKPOINTS: a folder of <name>.ckpt files whose <name>.pt references were written with
`python -m tests.golden.references --run-checkpoints FOLDER` before the change being checked.
"""

import os
from pathlib import Path
import pytest
from tests.conftest import fit_sold
from tests.golden import references
from tests.golden.test_golden import TOLERANCE
import torch

FOLDER = os.environ.get("SOLD_RUN_CHECKPOINTS")
CHECKPOINTS = sorted(Path(FOLDER).glob("*.ckpt")) if FOLDER else []

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not FOLDER, reason="SOLD_RUN_CHECKPOINTS is not set")]


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda path: path.stem)
def test_run_checkpoint_gives_the_same_outputs(checkpoint):
    expected = torch.load(checkpoint.with_suffix(".pt"))
    actual = references.run_checkpoint_outputs(checkpoint)
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.allclose(actual[key], expected[key], **TOLERANCE), key


@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda path: path.stem)
def test_run_checkpoint_resumes_training(checkpoint, tmp_path):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    num_optimizers = 5 if saved["hyper_parameters"]["finetune_autoencoder"] else 4
    if len(saved["optimizer_states"]) != num_optimizers:
        pytest.skip("saved before the autoencoder optimizer was checkpointed (BUGS_AND_PITFALLS.md #17)")
    # Two collection epochs of 25 steps: the first 50-step episode ends in the second, which trains. Small buffer and
    # batch and no evaluation, to run on a CPU.
    sold = references.load_sold(
        checkpoint,
        max_steps=saved["num_steps"] + 50,
        buffer_capacity=1000,
        batch_size=2,
        eval_freq=10**9,
        save_replay_buffer=False,
    )
    fit_sold(sold, tmp_path, checkpoint=checkpoint)
    assert sold.num_steps == saved["num_steps"] + 50
