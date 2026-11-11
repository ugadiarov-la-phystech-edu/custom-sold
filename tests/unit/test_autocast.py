"""bfloat16 autocast of SOLD's stages (OnlineModule.autocast_stages): mixed precision inside, float32 outside."""

from collections import defaultdict
import math
from pathlib import Path
import pytest
from tests import fakes
from tests.conftest import fit_sold, make_tiny_sold, read_metrics
from tests.golden import references
from tests.unit.test_compile import _assert_same_training, _episodes_of_actions, _steps, _two_environment_models
import torch

pytestmark = pytest.mark.usefixtures("eager_randomness")

STAGES = ["dynamics_loss", "reward_loss", "imagination", "actor_loss", "critic_loss", "act"]  # configs/train_sold.yaml


def _layer_output_dtypes(sold):
    """The dtypes the Linear and convolution layers of each model output from here on (SAVi's per part, e.g.
    "autoencoder.decoder")."""
    dtypes = defaultdict(set)
    for name, module in sold.named_modules():
        if isinstance(module, torch.nn.Linear | torch.nn.Conv2d | torch.nn.ConvTranspose2d):
            parts = name.split(".")
            model = ".".join(parts[:2]) if parts[0] == "autoencoder" else parts[0]
            module.register_forward_hook(lambda _, inputs, output, model=model: dtypes[model].add(output.dtype))
    return dtypes


def _record_stage_outputs(sold):
    """The tensors the stages return, recorded from here on."""
    outputs = []

    def flatten(output):
        return [t for value in output.values() for t in flatten(value)] if isinstance(output, dict) else [output]

    for stage in sold.STAGES:
        method = getattr(sold, f"_{stage}_stage")

        def recording(*args, method=method, **kwargs):
            output = method(*args, **kwargs)
            outputs.extend(flatten(output))
            return output

        setattr(sold, f"_{stage}_stage", recording)
    return outputs


def test_autocast_stages_compute_in_bfloat16_and_return_float32():
    sold, _ = _two_environment_models()
    sold.autocast_stages("bfloat16", STAGES)
    dtypes, outputs = _layer_output_dtypes(sold), _record_stage_outputs(sold)
    _steps(sold, 1)
    _episodes_of_actions(sold, torch.randint(0, 255, (8, 2, 3, 16, 16), dtype=torch.uint8), "train")

    bfloat16, float32 = {torch.bfloat16}, {torch.float32}
    for model in ["dynamics_predictor", "reward_predictor", "actor", "critic", "critic_target"]:
        assert dtypes[model] == bfloat16, model
    assert dtypes.pop("autoencoder.decoder") == bfloat16 | float32  # Predictions in bfloat16, reconstructions not.
    savi_parts = [model for model in dtypes if model.startswith("autoencoder.")]
    assert "autoencoder.encoder" in savi_parts and all(dtypes[model] == float32 for model in savi_parts)
    # Outside the stages everything is float32: their outputs, the parameters, gradients and optimizer states.
    assert outputs and {t.dtype for t in outputs if t.is_floating_point()} == float32
    gradients = [p.grad for p in sold.parameters() if p.grad is not None]
    assert gradients and {t.dtype for t in [*sold.parameters(), *gradients]} == float32
    states = [value for optimizer in sold.optimizers for state in optimizer.state.values() for value in state.values()]
    assert states and {value.dtype for value in states} == float32
    assert sold.return_moments.low.dtype == sold.return_moments.high.dtype == torch.float32


@pytest.mark.parametrize(
    "overrides",
    [{"finetune_autoencoder": False}, {"finetune_autoencoder": True, **references.EVERY_START}],
    ids=["frozen_autoencoder", "finetuning_every_start"],
)
def test_bfloat16_training_stays_close_to_float32(overrides):
    torch.manual_seed(0)
    float32 = make_tiny_sold(**overrides)
    torch.manual_seed(0)
    bfloat16 = make_tiny_sold(**overrides)
    bfloat16.autocast_stages("bfloat16", STAGES)
    _assert_same_training(_steps(bfloat16, 3), _steps(float32, 3), rtol=0.1, atol=0.02)


def test_compiled_autocast_stages_train_like_eager_ones(monkeypatch, aot_eager):
    torch.manual_seed(0)
    float32 = make_tiny_sold()
    models = {}
    for name in ["eager", "compiled"]:
        torch.manual_seed(0)
        models[name] = make_tiny_sold()
    models["compiled"].compile_stages("default")
    for sold in models.values():
        sold.autocast_stages("bfloat16", STAGES)
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    compiled = _steps(models["compiled"], 3)
    # AOTAutograd's decompositions round in other places than eager code: bfloat16 rounding errors of about 1%.
    _assert_same_training(compiled, _steps(models["eager"], 3), rtol=0.05, atol=1e-3)
    # The compiled stages are traced under the autocast: their losses are bfloat16's, not float32's.
    float32_losses = _steps(float32, 1)[0][0]
    assert any(not torch.equal(compiled[0][0][key], float32_losses[key]) for key in float32_losses)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_compiled_autocast_action_selection_equals_eager(mode, aot_eager):
    eager, compiled = _two_environment_models()
    compiled.compile_stages("default", ["encode", "act"])
    for sold in (eager, compiled):
        sold.autocast_stages("bfloat16", ["act"])
    obs = torch.randint(0, 255, (8, 2, 3, 16, 16), dtype=torch.uint8)
    for actual, reference in zip(_episodes_of_actions(compiled, obs, mode), _episodes_of_actions(eager, obs, mode)):
        assert actual.dtype == torch.float32 and torch.allclose(actual, reference, atol=1e-4)
        assert (actual >= eager.actor.lower_bound).all() and (actual <= eager.actor.upper_bound).all()


def test_autocast_stages_rejects_other_dtypes_and_unknown_stages():
    sold = make_tiny_sold()
    with pytest.raises(ValueError, match="Unsupported autocast dtype 'float16'"):
        sold.autocast_stages("float16")
    with pytest.raises(ValueError, match=r"Unknown stages \['policy'\]"):
        sold.autocast_stages("bfloat16", ["policy"])


def test_fit_with_autocast_keeps_float32_checkpoints(tmp_path, aot_eager):
    torch.manual_seed(0)
    sold = make_tiny_sold(max_steps=20)
    logger = fit_sold(
        sold,
        tmp_path / "bfloat16",
        compile_mode="default",
        compile_stages=["reward_loss", "act"],
        autocast_dtype="bfloat16",
        autocast_stages=STAGES,
    )
    assert "_dynamics_loss_stage" in vars(sold) and "_encode_stage" not in vars(sold)
    losses = [record["train/actor_loss"] for record in read_metrics(logger.log_dir) if "train/actor_loss" in record]
    assert losses and all(math.isfinite(loss) for loss in losses)
    checkpoint = sorted((tmp_path / "bfloat16").glob("logs/*/checkpoints/*.ckpt"))[-1]
    state_dict = torch.load(checkpoint, weights_only=False)["state_dict"]
    assert set(state_dict) == set(make_tiny_sold().state_dict())
    assert {value.dtype for value in state_dict.values() if value.is_floating_point()} == {torch.float32}
    resumed = make_tiny_sold(max_steps=30)  # Resumes in float32 from the bfloat16 run's checkpoint.
    fit_sold(resumed, tmp_path / "float32", checkpoint=checkpoint)
    assert resumed.num_steps > 20 and "_act_stage" not in vars(resumed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("finetune", [False, True], ids=["frozen_autoencoder", "finetuning"])
def test_cuda_graphs_with_autocast_train_like_eager_autocast(finetune):
    from torch._dynamo.utils import counters

    models = {}
    for name in ["eager", "graphed"]:
        torch.manual_seed(0)
        models[name] = make_tiny_sold(finetune_autoencoder=finetune).to("cuda")
        models[name].fused_optimizers = True
    models["graphed"].compile_stages("reduce-overhead")
    for sold in models.values():
        sold.autocast_stages("bfloat16", STAGES)
    # Four steps: compiled code warms up, records the CUDA graphs, then replays them. Inductor keeps fused bfloat16
    # intermediates in float32, eager code rounds each op's output: gradients differ by up to about 8%.
    graphed, eager = (_steps(models[name], 4, device="cuda") for name in ["graphed", "eager"])
    _assert_same_training(graphed, eager, rtol=0.15, atol=1e-3)
    assert counters["inductor"]["cudagraph_skips"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fit_with_cuda_graphs_and_autocast_logs_metrics_and_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(torch._inductor.config, "fallback_random", False)  # As in training.
    sold = make_tiny_sold(fakes.FakeBatchEnv(2), max_steps=30, save_replay_buffer=False)
    logger = fit_sold(
        sold, tmp_path, device="cuda", compile_mode="reduce-overhead", autocast_dtype="bfloat16", autocast_stages=STAGES
    )
    records = read_metrics(logger.log_dir)
    losses = [record["train/actor_loss"] for record in records if "train/actor_loss" in record]
    assert len(losses) == 4 and len(set(losses)) == 4 and all(math.isfinite(loss) for loss in losses)
    assert list(Path(logger.log_dir).glob("**/latent_imagination-step=*.png"))
