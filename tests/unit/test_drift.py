"""utils/drift.py: comparing the training updates of models that should train alike."""

import math
import numpy as np
import pytest
from sold.utils import drift
from tests.conftest import make_tiny_sold
from tests.golden import references
import torch


def _model(seed=0, device="cpu", **overrides):
    torch.manual_seed(seed)
    sold = make_tiny_sold(**overrides).to(device).train()
    sold.optimizers = sold.configure_optimizers()
    return sold


def _batches(sold, num_batches, device="cpu"):
    images, actions, rewards = (x.to(device) for x in references._tiny_sold_batch(sold))
    return [{"obs": images * 255, "action": actions, "reward": rewards}] * num_batches


def _errors(record, model):
    prefix = f"{model}/err/"
    return {key[len(prefix) :]: value for key, value in record.items() if key.startswith(prefix)}


def _optimizer_states(model):
    return [state for optimizer in model.optimizers for state in optimizer.state.values()]


def _pointers(model):
    tensors = [*model.state_dict().values(), *(v for state in _optimizer_states(model) for v in state.values())]
    return [tensor.data_ptr() for tensor in tensors]


def test_copy_training_state_copies_into_the_existing_tensors():
    source, target, fresh = _model(0), _model(1), _model(2)
    for model in (source, target):
        drift.seed_everything(0)
        model.training_step(_batches(model, 1)[0])  # Creates the optimizer states.
    pointers = _pointers(target)

    drift.copy_training_state(source, target)
    drift.copy_training_state(source, fresh)  # Without optimizer states yet: they are cloned.

    assert _pointers(target) == pointers
    assert not set(_pointers(fresh)) & set(_pointers(source))
    for copy in (target, fresh):
        for key, value in source.state_dict().items():
            assert torch.equal(copy.state_dict()[key], value), key
        for source_state, state in zip(_optimizer_states(source), _optimizer_states(copy), strict=True):
            assert all(torch.equal(state[key], value) for key, value in source_state.items())


def test_identical_models_have_no_errors():
    eager, repeat = _model(), _model(1)  # Re-synced from the first step on, so their initial weights do not matter.
    records = list(drift.lockstep({"E": eager, "E2": repeat}, _batches(eager, 2)))
    errors = _errors(records[0], "E2")
    assert {"stage/dynamics_loss/dynamics_loss", "grad/actor", "update/actor", "state/critic_target"} <= set(errors)
    assert "buffer/return_moments.low" in errors and "update/autoencoder" in errors  # Frozen: no update, no error.
    assert all(value == 0 for record in records for value in _errors(record, "E2").values())
    high = records[1]["E/buffer/return_moments.high"]
    assert records[1]["step"] == 1 and high == records[1]["E2/buffer/return_moments.high"] and high != 0
    assert not any(entry["flagged"] for entry in drift.summarize(records, "E2").values())


def test_compiled_stages_match_eager_ones_update_by_update(aot_eager, eager_randomness):
    eager, compiled = _model(), _model()
    compiled.compile_stages("default")
    records = list(drift.lockstep({"E": eager, "C": compiled}, _batches(eager, 3)))
    assert max(max(_errors(record, "C").values()) for record in records) < 1e-3
    assert not any(entry["flagged"] for entry in drift.summarize(records, "C").values())


def test_a_buffer_the_update_does_not_change_is_flagged():
    eager, broken = _model(), _model()

    def percentiles_without_update(x):  # Normalizes with the old percentiles and never updates them.
        low, high = broken.return_moments.low, broken.return_moments.high
        return low.clone(), torch.clamp(high - low, min=1)

    broken.return_moments.forward = percentiles_without_update
    summary = drift.summarize(list(drift.lockstep({"E": eager, "B": broken}, _batches(eager, 2))), "B")
    high = summary["buffer/return_moments.high"]
    assert high["flagged"] and high["step_of_max"] == 0


def test_a_stale_stage_output_is_flagged():
    eager, stale = _model(), _model()
    reward_loss_stage, previous = stale._reward_loss_stage, []

    def stale_reward_loss(slots, rewards):  # From the second step on, returns the first step's outputs.
        outputs = reward_loss_stage(slots, rewards)
        previous.append({key: value.detach() for key, value in outputs.items()})
        return outputs if len(previous) == 1 else previous[0] | {"reward_loss": outputs["reward_loss"]}

    stale._reward_loss_stage = stale_reward_loss
    batches = [batch | {"reward": batch["reward"] * (i + 1)} for i, batch in enumerate(_batches(eager, 3))]
    summary = drift.summarize(list(drift.lockstep({"E": eager, "S": stale}, batches)), "S")
    stale_metric = summary["stage/reward_loss/reward_mse_loss"]
    assert stale_metric["flagged"] and stale_metric["step_of_max"] > 0
    assert summary["stage/reward_loss/reward_loss"]["max"] < 1e-3  # Computed afresh.


def test_free_running_models_with_other_random_numbers_drift_apart():
    eager, other = _model(), _model()
    records = list(drift.lockstep({"E": eager, "O": other}, _batches(eager, 3), resync=False, seed_offsets={"O": 100}))
    assert all(record["O/err/state/actor"] > 0 for record in records)
    assert records[-1]["O/err/state/dynamics_predictor"] > records[0]["O/err/state/dynamics_predictor"]


def test_summarize_and_bias():
    records = [{"step": float(step), "A/err/x": 1e-4, "F/err/x": 2e-3, "A/x": 1.0, "R/x": 0.0} for step in range(8)]
    records[5]["A/err/x"] = 0.05
    summary = drift.summarize(records, "A", floor_model="F")["x"]
    assert summary["max"] == 0.05 and summary["step_of_max"] == 5 and summary["floor"] == 2e-3 and summary["flagged"]
    assert summary["median"] == 1e-4 and 1e-4 < summary["p99"] <= 0.05
    assert not drift.summarize(records, "A", floor_model="F", threshold=0.1)["x"]["flagged"]
    growing = [{"step": float(step), "A/err/x": 10.0 ** (step / 2) * 1e-6} for step in range(8)]  # Up to 3e-3.
    summary = drift.summarize(growing, "A")["x"]
    assert summary["max"] < 1e-2 and summary["flagged"]  # Below the threshold, but growing.

    assert drift.bias(records, "x", "A", "R") == {"mean": 1.0, "standard_error": 0.0, "z": 0.0}
    noise = np.random.default_rng(0).normal(0.5, 1.0, 400)
    independent = drift.bias([{"A/x": value, "R/x": 0.0} for value in noise], "x", "A", "R")
    assert independent["mean"] == pytest.approx(0.5, abs=0.15) and independent["z"] > 5
    blocks = [{"A/x": float(step >= 50), "R/x": 0.0} for step in range(100)]  # Strongly autocorrelated.
    assert drift.bias(blocks, "x", "A", "R")["z"] < 3  # Naively about 10.


def test_relative_error():
    assert drift.relative_error(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("nan")])) == 0
    assert drift.relative_error(torch.zeros(2), torch.zeros(2)) == 0
    assert drift.relative_error(torch.ones(2), torch.zeros(2)) == math.inf
    assert drift.relative_error(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 5.0])) == pytest.approx(math.sqrt(10) / 5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graph_updates_match_eager_ones_update_by_update(eager_randomness):
    from torch._dynamo.utils import counters

    eager, graphed = _model(device="cuda"), _model(device="cuda")
    graphed.compile_stages("reduce-overhead")
    # Eight steps: compiled code warms up, records the CUDA graphs, then replays them.
    records = list(drift.lockstep({"E": eager, "G": graphed}, _batches(eager, 8, "cuda")))
    worst = {key: max(_errors(record, "G")[key] for record in records) for key in _errors(records[0], "G")}
    # Decoded per-slot images of predicted slots amplify rounding most (about 2e-3); losses and gradients < 1e-3.
    assert max(worst.values()) < 1e-2, sorted(worst.items(), key=lambda item: -item[1])[:8]
    assert max(value for key, value in worst.items() if key.endswith("_loss") or key.startswith("grad/")) < 1e-3
    assert counters["inductor"]["cudagraph_skips"] == 0
