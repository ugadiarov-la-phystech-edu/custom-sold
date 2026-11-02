"""torch.compile of SOLD's training stages (OnlineModule.compile_stages) and the CUDA-graph step marker."""

from pathlib import Path
import pytest
from tests import fakes
from tests.conftest import fit_sold, make_tiny_sold, read_metrics
from tests.golden import references
import torch


@pytest.fixture(autouse=True)
def eager_randomness(monkeypatch):
    """Compiled code draws the same random numbers as eager code (dropout, rsample), so their results compare."""
    torch._dynamo.reset()
    monkeypatch.setattr(torch._inductor.config, "fallback_random", True)
    yield
    torch._dynamo.reset()


def _steps(sold, num_steps, device="cpu"):
    """The losses of num_steps training steps on a fixed batch, and the gradients after the first step.

    Later parameters are compared through the losses: Adam turns rounding noise in gradients that are zero in theory
    (e.g. of attention key biases) into steps of about the learning rate.
    """
    sold.to(device).train()
    sold.optimizers = sold.configure_optimizers()
    images, actions, rewards = (x.to(device) for x in references._tiny_sold_batch(sold))
    losses, gradients = [], {}
    for step in range(num_steps):
        torch.manual_seed(step)
        if sold.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        outputs = sold.training_step({"obs": images * 255, "action": actions, "reward": rewards})
        losses.append({key: value.detach().clone() for key, value in outputs.items() if key.endswith("_loss")})
        if step == 0:
            gradients = {name: p.grad.clone() for name, p in sold.named_parameters() if p.grad is not None}
    return losses, gradients


def _assert_same_training(actual, expected):
    actual_losses, actual_gradients = actual
    expected_losses, expected_gradients = expected
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, reference in expected_gradients.items():  # Relative to the norm: summation orders differ.
        assert (actual_gradients[name] - reference).norm() <= 1e-3 * reference.norm() + 1e-6, name
    for step, (losses, reference) in enumerate(zip(actual_losses, expected_losses)):
        for key in reference:
            assert torch.allclose(losses[key], reference[key], rtol=1e-3, atol=1e-5), f"step {step}: {key}"


@pytest.fixture
def aot_eager(monkeypatch):
    """Compile with the aot_eager backend: dynamo and AOTAutograd as in production (graph breaks, recompiles,
    autograd), with eager kernels instead of Inductor's, which take minutes to build on a CPU. The GPU tests below
    cover Inductor and CUDA graphs."""
    compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, mode=None, **kwargs: compile(fn, backend="aot_eager", **kwargs))


@pytest.mark.parametrize(
    "overrides",
    [{"finetune_autoencoder": False}, {"finetune_autoencoder": True, **references.EVERY_START}],
    ids=["frozen_autoencoder", "finetuning_every_start"],
)
def test_compiled_stages_train_like_eager_ones(overrides, monkeypatch, aot_eager):
    torch.manual_seed(0)
    eager = make_tiny_sold(**overrides)
    torch.manual_seed(0)
    compiled = make_tiny_sold(**overrides)
    compiled.compile_stages("default")
    assert not compiled.cudagraphs
    if "num_context" not in overrides:  # A fixed number of context frames compiles each stage once.
        monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    _assert_same_training(_steps(compiled, 3), _steps(eager, 3))


def test_compile_stages_compiles_only_the_given_stages():
    sold = make_tiny_sold()
    eager_method = sold._reward_loss_stage
    sold.compile_stages("reduce-overhead", ["imagination", "actor_loss"])
    assert sold.cudagraphs
    assert {"_imagination_stage", "_actor_loss_stage"} <= set(vars(sold))  # Compiled versions shadow the methods.
    assert "_reward_loss_stage" not in vars(sold) and sold._reward_loss_stage == eager_method
    with pytest.raises(ValueError, match=r"Unknown stages \['policy'\]"):
        sold.compile_stages("default", ["policy"])


def test_fit_with_compiled_stages_keeps_checkpoints_compatible(tmp_path, aot_eager):
    torch.manual_seed(0)
    sold = make_tiny_sold(max_steps=20)
    stages = ["reward_loss", "critic_loss"]
    logger = fit_sold(sold, tmp_path / "compiled", compile_mode="default", compile_stages=stages)
    assert "_critic_loss_stage" in vars(sold)
    assert "timing/learn/critic_loss_ms" in set().union(*read_metrics(logger.log_dir))
    checkpoint = sorted((tmp_path / "compiled").glob("logs/*/checkpoints/*.ckpt"))[-1]
    assert set(torch.load(checkpoint, weights_only=False)["state_dict"]) == set(make_tiny_sold().state_dict())
    eager = make_tiny_sold(max_steps=30)  # Resumes eagerly from the compiled run's checkpoint.
    fit_sold(eager, tmp_path / "eager", checkpoint=checkpoint)
    assert eager.num_steps > 20 and "_critic_loss_stage" not in vars(eager)


def test_updates_and_action_selection_mark_cuda_graph_steps(tmp_path, monkeypatch):
    marks, selections = [], []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: marks.append(True))
    sold = make_tiny_sold(max_steps=20, num_updates=2, save_replay_buffer=False)
    sold.cudagraphs = True  # As compile_stages sets it for CUDA-graph modes; the stages stay eager here.
    select_action = sold.select_action

    def counting_select_action(*args, **kwargs):
        selections.append(1)
        return select_action(*args, **kwargs)

    monkeypatch.setattr(sold, "select_action", counting_select_action)
    fit_sold(sold, tmp_path)
    # Two rounds of two updates after the seed steps (10, then 5 + 5 steps), and every action selection.
    assert len(marks) == 2 * 2 + len(selections) and selections


def _episodes_of_actions(sold, obs, mode, episodes=1):
    """The actions of whole episodes (steps along obs's first dimension) chosen by sold._policy."""
    actions = []
    for _ in range(episodes):
        torch.manual_seed(0)
        actions += [sold._policy(obs[t], is_first=t == 0, mode=mode).clone() for t in range(obs.shape[0])]
    return actions


def _two_environment_models(device="cpu"):
    torch.manual_seed(0)
    eager = make_tiny_sold(fakes.FakeBatchEnv(2)).to(device)
    torch.manual_seed(0)
    return eager, make_tiny_sold(fakes.FakeBatchEnv(2)).to(device)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_compiled_action_selection_equals_eager(mode, monkeypatch, aot_eager):
    eager, compiled = _two_environment_models()
    compiled.compile_stages("default", ["encode", "act"])
    obs = torch.randint(0, 255, (8, 2, 3, 16, 16), dtype=torch.uint8)  # A whole episode of two environments.
    expected = _episodes_of_actions(eager, obs, mode)
    _episodes_of_actions(compiled, obs, mode)  # Compiles: one graph for the first step's encoding, one for the rest.
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    for actual, reference in zip(_episodes_of_actions(compiled, obs, mode), expected):
        assert torch.allclose(actual, reference, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graph_action_selection_equals_eager():
    from torch._dynamo.utils import counters

    eager, graphed = _two_environment_models("cuda")
    graphed.compile_stages("reduce-overhead", ["encode", "act"])
    obs = torch.randint(0, 255, (8, 2, 3, 16, 16), dtype=torch.uint8)
    # Three episodes: compiled code warms up, records the CUDA graphs, then replays them.
    expected = _episodes_of_actions(eager, obs, "eval", 3)
    for actual, reference in zip(_episodes_of_actions(graphed, obs, "eval", 3), expected):
        assert torch.allclose(actual, reference, atol=1e-4)
    assert counters["inductor"]["cudagraph_skips"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("finetune", [False, True], ids=["frozen_autoencoder", "finetuning"])
def test_cuda_graphs_train_like_eager_stages(finetune):
    from torch._dynamo.utils import counters

    torch.manual_seed(0)
    eager = make_tiny_sold(finetune_autoencoder=finetune)
    torch.manual_seed(0)
    graphed = make_tiny_sold(finetune_autoencoder=finetune)
    graphed.fused_optimizers = eager.fused_optimizers = True
    graphed.compile_stages("reduce-overhead")
    # Four steps: compiled code warms up, records the CUDA graphs, then replays them.
    _assert_same_training(_steps(graphed, 4, device="cuda"), _steps(eager, 4, device="cuda"))
    assert counters["inductor"]["cudagraph_skips"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fit_with_cuda_graphs_logs_metrics_and_figures(tmp_path, monkeypatch):
    # Inductor's own random numbers, as in training: the eager normal(mean, std) checks std on the host, which a CUDA
    # graph cannot record.
    monkeypatch.setattr(torch._inductor.config, "fallback_random", False)
    sold = make_tiny_sold(fakes.FakeBatchEnv(2), max_steps=30, save_replay_buffer=False)  # Action selection too.
    logger = fit_sold(sold, tmp_path, device="cuda", compile_mode="reduce-overhead")
    records = read_metrics(logger.log_dir)
    losses = [record["train/actor_loss"] for record in records if "train/actor_loss" in record]
    assert len(losses) == 4 and len(set(losses)) == 4  # Each round's own value, not an overwritten graph output.
    assert list(Path(logger.log_dir).glob("**/latent_imagination-step=*.png"))  # Figures after the second evaluation.
