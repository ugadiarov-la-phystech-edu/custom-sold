"""train_sold.SOLDModule: construction, losses, imagination and action selection."""

import copy
import numpy as np
from pathlib import Path
import pytest
from sold import rl
from sold.modeling.autoencoder.losses import reconstruction_loss
from sold.train_sold import OptimizerSettings, SOLDModule, optimization_settings
from tests import fakes
from tests.conftest import fit_sold, make_tiny_sold, read_metrics, tiny_sold_kwargs
import torch
import torch.nn.functional as F


class LogRecorder:
    def __init__(self):
        self.names = []

    def __call__(self, name, value, *args, **kwargs):
        self.names.append(name)


def _batch(sold, batch_size=2):
    length = sold.sequence_length
    images = torch.rand(batch_size, length, 3, 16, 16)
    actions = torch.rand(batch_size, length, 2) * 2 - 1
    actions[:, 0] = float("nan")
    rewards = torch.rand(batch_size, length)
    rewards[:, 0] = float("nan")
    return images, actions, rewards


def test_construction(tiny_sold):
    assert (tiny_sold.min_num_context, tiny_sold.max_num_context) == (2, 2)
    assert tiny_sold.sequence_length == 5
    assert torch.allclose(tiny_sold.discounts, 0.96 ** torch.arange(3.0).unsqueeze(0))
    assert "env" not in tiny_sold.hparams and tiny_sold.hparams["imagination_horizon"] == 3
    assert tiny_sold.critic_target is not tiny_sold.critic
    for a, b in zip(tiny_sold.critic.parameters(), tiny_sold.critic_target.parameters()):
        assert torch.equal(a, b)
    assert tiny_sold.dynamics_predictor.input_buffer_size == 5
    assert torch.equal(tiny_sold.actor.lower_bound, torch.tensor([-1.0, -1.0]))


def _actor_loss(sold, imagination):
    return sold.compute_actor_loss(
        imagination.lambda_returns,
        imagination.predicted_values_targ,
        imagination.action_log_probs,
        imagination.action_entropies,
    )


def test_critic_target_is_not_trained_by_gradients(tiny_sold):
    assert not any(p.requires_grad for p in tiny_sold.critic_target.parameters())
    images, actions, _ = _batch(tiny_sold)
    slots = tiny_sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    imagination = tiny_sold.imagine_ahead(slots, actions)
    assert not imagination.predicted_values_targ.requires_grad
    _actor_loss(tiny_sold, imagination)["actor_loss"].backward()
    assert all(p.grad is None for p in tiny_sold.critic_target.parameters())


def test_imagine_ahead_with_batch_size_one(tiny_sold):
    images, actions, _ = _batch(tiny_sold, batch_size=1)
    slots = tiny_sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    imagination = tiny_sold.imagine_ahead(slots, actions)
    assert (
        imagination.lambda_returns.shape
        == imagination.predicted_values_targ.shape
        == imagination.action_log_probs.shape
        == (1, 3)
    )


def test_optimizer_settings_are_grouped_per_model(tiny_sold):
    assert tiny_sold.optimizer_settings["dynamics"] == OptimizerSettings(learning_rate=1e-3, grad_clip=3.0)
    assert tiny_sold.optimizer_settings["autoencoder"].grad_clip == 0.05
    critic = {"learning_rate": 1e-3, "grad_clip": 10.0, "warmup_updates": 0}
    assert tiny_sold.hparams["optimization"]["critic"] == critic


def test_flat_optimizer_settings_of_old_checkpoints_are_grouped():
    kwargs = tiny_sold_kwargs(fakes.make_fake_env())
    flat = {
        f"{model}_{setting}": value
        for model, settings in kwargs.pop("optimization").items()
        for setting, value in settings.items()
    }
    sold = SOLDModule(optimization=None, **kwargs, **flat)
    assert sold.optimizer_settings["reward"] == OptimizerSettings(learning_rate=1e-3, grad_clip=10.0)
    assert "dynamics_learning_rate" not in sold.hparams  # Checkpoints store the grouped settings.
    assert sold.hparams["optimization"]["dynamics"] == {"learning_rate": 1e-3, "grad_clip": 3.0, "warmup_updates": 0}
    sold = SOLDModule(optimization=None, **kwargs, **flat, autoencoder_warmup_updates=7)
    assert sold.optimizer_settings["autoencoder"].warmup_updates == 7


def test_optimization_settings_reject_unknown_arguments_and_settings():
    optimization = tiny_sold_kwargs(fakes.make_fake_env())["optimization"]
    with pytest.raises(TypeError, match="Unexpected argument: foo_rate"):
        optimization_settings(optimization, {"foo_rate": 1e-3})
    with pytest.raises(TypeError, match="learnin_rate"):  # A misspelled setting.
        optimization_settings({**optimization, "actor": {"learnin_rate": 1e-3, "grad_clip": 1.0}}, {})


def test_context_range():
    sold = make_tiny_sold(num_context=[1, 3])
    assert (sold.min_num_context, sold.max_num_context, sold.sequence_length) == (1, 3, 6)
    with pytest.raises(ValueError, match="min_num_context"):
        make_tiny_sold(num_context=(3, 1))


def test_configure_optimizers(tiny_sold):
    optimizers = tiny_sold.configure_optimizers()
    assert [o.param_groups[0]["lr"] for o in optimizers] == [1e-3] * 4
    owners = [tiny_sold.dynamics_predictor, tiny_sold.reward_predictor, tiny_sold.actor, tiny_sold.critic]
    for optimizer, owner in zip(optimizers, owners):
        assert optimizer.param_groups[0]["params"] == list(owner.parameters())


def test_compute_lambda_returns_uses_the_configured_discount_and_lambda():
    sold = make_tiny_sold(discount_factor=0.9, return_lambda=0.5)
    rewards, values = torch.randn(4, 3), torch.randn(4, 3)
    assert torch.equal(
        sold.compute_lambda_returns(rewards, values), rl.returns.lambda_returns(rewards, values, 0.9, 0.5)
    )


@pytest.mark.parametrize("gradients", ["dynamics", "reinforce"])
def test_compute_actor_loss_normalizes_with_the_return_moments(gradients):
    sold = make_tiny_sold(actor_gradients=gradients, actor_entropy_loss_weight=0.01)
    returns, values, log_probs, entropies = torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.rand(2, 3)
    moments = copy.deepcopy(sold.return_moments)
    losses = sold.compute_actor_loss(returns, values, log_probs, entropies)
    expected = rl.losses.actor_loss(
        returns, values, log_probs, entropies, sold.discounts, *moments(returns[:, :-1]), gradients, 0.01
    )
    assert losses.keys() == expected.keys() and all(torch.equal(losses[k], expected[k]) for k in losses)
    assert torch.equal(sold.return_moments.low, moments.low)  # The running percentiles were updated.


@pytest.mark.parametrize("overrides, weight", [({}, 0.1), ({"critic_regularization_weight": 0.5}, 0.5)])
def test_compute_critic_loss_uses_the_configured_regularization_weight(overrides, weight):
    sold = make_tiny_sold(**overrides)  # The value hard-coded before it was configurable.
    imagined_slots, returns, targets = torch.randn(2, 5, 2, 8), torch.randn(2, 3), torch.randn(2, 3)
    losses = sold.compute_critic_loss(imagined_slots, 2, returns, targets)
    # The critic predicts values from the context's last step on, for all but the last imagined step.
    dist = sold.critic(imagined_slots[:, :-1], start=1)
    assert dist.mean.shape == (2, 3, 1)
    expected = rl.losses.critic_loss(dist, returns, targets, sold.discounts, weight)
    assert losses.keys() == expected.keys()
    assert all(torch.allclose(torch.as_tensor(losses[k]), torch.as_tensor(expected[k])) for k in losses)


def test_compute_dynamics_loss(tiny_sold):
    images, actions, _ = _batch(tiny_sold)
    slots = tiny_sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    losses = tiny_sold.compute_dynamics_loss(images, slots, actions)
    assert torch.allclose(losses["dynamics_loss"], losses["slot_loss"] + losses["image_loss"])
    assert tiny_sold.dynamics_predictor.training


def test_compute_reward_loss_ignores_first_step(tiny_sold):
    images, actions, rewards = _batch(tiny_sold)
    outputs = tiny_sold.autoencoder(images, actions[:, 1:])
    losses = tiny_sold.compute_reward_loss(images, outputs["reconstructions"], outputs["slots"], rewards)
    assert torch.isfinite(losses["reward_loss"]) and losses["reward_mse_loss"].dim() == 0
    assert not losses["reward_mse_loss"].requires_grad
    # The masked means equal the means over the steps after the first.
    valid = ~torch.isnan(rewards)
    log_probs = tiny_sold.reward_predictor(outputs["slots"]).log_prob(torch.nan_to_num(rewards).unsqueeze(2))
    assert torch.allclose(losses["reward_loss"], -log_probs.reshape(valid.shape)[valid].mean())
    expected_mse = F.mse_loss(losses["predicted_rewards"][valid], rewards[valid])
    assert torch.allclose(losses["reward_mse_loss"], expected_mse)


@pytest.mark.parametrize("from_every", [False, True])
def test_imagine_ahead_shapes(from_every):
    sold = make_tiny_sold(start_imagination_from_every=from_every)
    images, actions, _ = _batch(sold)
    slots = sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    imagination = sold.imagine_ahead(slots, actions)
    returns = imagination.lambda_returns
    batch = 2 * (5 - 2 + 1) if from_every else 2
    assert returns.shape == imagination.predicted_values_targ.shape == imagination.action_log_probs.shape == (batch, 3)
    # The entropies broadcast against the discounts (1, 3) in the actor loss (BUGS_AND_PITFALLS.md #26).
    assert imagination.action_entropies.shape == (batch, 3, 1)
    assert imagination.imagined_slots.shape == (batch, 2 + 3, 2, 8) and not imagination.imagined_slots.requires_grad
    assert not sold.dynamics_predictor.training
    assert all(p.requires_grad for p in sold.critic.parameters())  # Unfrozen again.
    returns.mean().backward()
    assert any(p.grad is not None for p in sold.actor.parameters())
    assert all(p.grad is None for p in sold.critic.parameters())


def _log_visualizations(sold):
    """Compute one batch's losses as training_step does and log the figures; returns the logged names."""
    recorder = LogRecorder()
    sold.log = sold.log_image = recorder
    images, actions, rewards = _batch(sold)
    reconstruction = reconstruction_loss(sold.autoencoder, images, actions)
    slots = reconstruction["slots"].detach()
    outputs = sold.compute_dynamics_loss(images, slots, actions)
    outputs |= sold.compute_reward_loss(images, reconstruction["reconstructions"], slots, rewards)
    assert recorder.names == [], "the loss functions only compute"
    sold.log_visualizations(rewards, reconstruction, outputs, sold.imagine_ahead(slots, actions))
    return recorder.names


def test_visualizations_after_evaluation(tiny_sold):
    assert _log_visualizations(tiny_sold) == [
        "autoencoder_reconstruction",
        "dynamics_prediction",
        "reward_prediction",
        "reward_predictor_attention",
        "latent_imagination",
        "actor_attention",
    ]


def test_visualizations_without_slot_masks():
    import functools
    from sold.modeling.autoencoder.cnn import CnnDecoder, CnnEncoder
    from sold.modeling.autoencoder.cnn.autoencoder import Cnn

    autoencoder = Cnn(
        CnnEncoder([4, 4], [3, 3], [2, 2]),
        functools.partial(CnnDecoder, image_size=(16, 16), num_channels=[4, 4], kernel_sizes=[3, 3], strides=[2, 2]),
    )
    names = _log_visualizations(make_tiny_sold(autoencoder=autoencoder))
    # No attention figures without slot masks.
    assert names == ["autoencoder_reconstruction", "dynamics_prediction", "reward_prediction", "latent_imagination"]


def test_training_step_logs_figures_only_after_evaluation(tiny_sold, monkeypatch):
    calls = []
    monkeypatch.setattr(tiny_sold, "log_visualizations", lambda *args: calls.append(args))
    monkeypatch.setattr(tiny_sold, "_optimize", lambda *args: None)
    tiny_sold.optimizers = [None] * 4
    images, actions, rewards = _batch(tiny_sold)
    batch = {"obs": images * 255, "action": actions, "reward": rewards}
    for visualize in [False, True]:
        outputs = tiny_sold.training_step(batch, visualize=visualize)
    assert len(calls) == 1 and {"dynamics_loss", "reward_loss", "actor_loss", "critic_loss"} <= set(outputs)


def test_select_action_modes(tiny_sold):
    obs = torch.randint(0, 255, (1, 3, 16, 16), dtype=torch.uint8)  # A batch of one environment.
    random_action = tiny_sold.select_action(obs, is_first=True, mode="random")
    assert random_action.shape == (1, 2) and tiny_sold._slot_history.shape == (1, 1, 2, 8)
    train_action = tiny_sold.select_action(obs, mode="train")
    assert tiny_sold._slot_history.shape == (1, 2, 2, 8)
    eval_a = tiny_sold.select_action(obs, is_first=True, mode="eval")
    eval_b = tiny_sold.select_action(obs, is_first=True, mode="eval")
    assert torch.equal(eval_a, eval_b) and tiny_sold._slot_history.shape == (1, 1, 2, 8)
    for action in (random_action, train_action, eval_a):
        assert action.abs().max() <= 1.0 and not action.requires_grad
    with pytest.raises(ValueError, match="Invalid mode"):
        tiny_sold.select_action(obs, mode="greedy")


def test_select_action_clamps_with_first_dimension_bounds():
    env = fakes.make_fake_env()
    env.unwrapped.action_space = env.action_space = type(env.action_space)(
        low=-0.1, high=0.1, shape=(2,), dtype=env.action_space.dtype
    )
    sold = make_tiny_sold(env)
    obs = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
    for _ in range(5):
        assert sold.select_action(obs, is_first=True, mode="train").abs().max() <= 0.1


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #26: the entropy bonus ignores per-step discounts")
def test_entropy_bonus_uses_the_discount_of_its_own_step():
    sold = make_tiny_sold()
    images = torch.rand(2, sold.sequence_length, 3, 16, 16)
    actions = torch.rand(2, sold.sequence_length, 2)
    slots = sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    imagination = sold.imagine_ahead(slots, actions)
    assert imagination.action_entropies.shape == (2, 3, 1)  # One entropy per (sample, step), as imagined.
    entropies = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unsqueeze(-1)
    losses = sold.compute_actor_loss(
        imagination.lambda_returns, imagination.predicted_values_targ, imagination.action_log_probs, entropies
    )
    expected = -sold.actor_entropy_loss_weight * torch.mean(sold.discounts * entropies.squeeze(-1))
    assert torch.allclose(losses["actor_entropy_loss"], expected), "each entropy is weighted by the mean discount"


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #29: imagined actions are not limited to the action space")
def test_imagined_actions_stay_within_the_action_space():
    sold = make_tiny_sold()
    with torch.no_grad():
        sold.actor.mlp[-1].bias[2:].fill_(10.0)  # Maximal standard deviation (1.0), as early in training.
    images = torch.rand(4, sold.sequence_length, 3, 16, 16)
    actions = torch.rand(4, sold.sequence_length, 2) * 2 - 1
    slots = sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    used = []
    original = sold.dynamics_predictor.predict_slots
    sold.dynamics_predictor.predict_slots = lambda slots, actions, **kwargs: (
        used.append(actions[:, -1].detach()),
        original(slots, actions, **kwargs),
    )[1]
    sold.imagine_ahead(slots, actions)
    imagined = torch.cat(used)
    assert imagined.abs().max() <= 1.0, f"dynamics model receives actions up to {imagined.abs().max():.2f}"


# BUGS_AND_PITFALLS.md #17 (fixed).
def test_actions_are_clamped_per_dimension():
    env = fakes.make_fake_env()
    env.unwrapped.action_space = env.action_space = type(env.action_space)(
        low=np.array([-1.0, -0.1], np.float32), high=np.array([1.0, 0.1], np.float32)
    )
    sold = make_tiny_sold(env)
    obs = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
    actions = torch.cat([sold.select_action(obs, is_first=True, mode="train") for _ in range(50)])
    assert actions[:, 1].abs().max() <= 0.1 + 1e-6


def _warmup_sold(warmup_updates, **overrides):
    optimization = tiny_sold_kwargs(fakes.make_fake_env())["optimization"]
    optimization["autoencoder"] = {**optimization["autoencoder"], "warmup_updates": warmup_updates}
    sold = make_tiny_sold(finetune_autoencoder=True, optimization=optimization, **overrides)
    sold.optimizers = sold.configure_optimizers()
    return sold


def _autoencoder_update(sold):
    autoencoder_optimizer = sold.optimizers[-1]
    loss = sum(param.square().sum() for param in sold.autoencoder.parameters())
    sold._optimize("autoencoder", autoencoder_optimizer, loss)
    return autoencoder_optimizer.param_groups[0]["lr"]


def test_warmup_raises_the_learning_rate_linearly_after_a_resume():
    sold = _warmup_sold(warmup_updates=4)
    sold.num_updates_since_restore = 0
    rates = []
    for update in range(6):
        sold.num_updates_since_restore = update
        rates.append(_autoencoder_update(sold))
    assert rates == pytest.approx([0.25e-3, 0.5e-3, 0.75e-3, 1e-3, 1e-3, 1e-3])
    assert sold.learning_rates == {"autoencoder": 1e-3}


def test_no_warmup_without_a_resume_or_warmup_updates():
    assert _autoencoder_update(_warmup_sold(warmup_updates=4)) == 1e-3  # num_updates_since_restore is None.
    sold = _warmup_sold(warmup_updates=0)
    sold.num_updates_since_restore = 0
    assert _autoencoder_update(sold) == 1e-3 and sold.learning_rates == {}


def test_resumed_training_warms_up_and_logs_the_learning_rate(tmp_path):
    sold = _warmup_sold(warmup_updates=10**6, max_steps=20)
    logger = fit_sold(sold, tmp_path / "first")
    checkpoint = next((Path(logger.log_dir) / "checkpoints").glob("*.ckpt"))
    resumed = _warmup_sold(warmup_updates=10**6, max_steps=40)
    logger = fit_sold(resumed, tmp_path / "second", checkpoint=checkpoint)
    records = read_metrics(logger.log_dir)
    rates = [record["learning_rate/autoencoder"] for record in records if "learning_rate/autoencoder" in record]
    assert rates and all(0 < rate < 1e-6 for rate in rates) and rates == sorted(rates)
    assert all("learning_rate/dynamics" not in record for record in records)


# BUGS_AND_PITFALLS.md #17 (fixed).
def test_autoencoder_optimizer_state_is_checkpointed(tmp_path):
    sold = make_tiny_sold(finetune_autoencoder=True, max_steps=20)
    logger = fit_sold(sold, tmp_path)
    checkpoint = torch.load(next((Path(logger.log_dir) / "checkpoints").glob("*.ckpt")), weights_only=False)
    saved = sum(len(group["params"]) for state in checkpoint["optimizer_states"] for group in state["param_groups"])
    expected = sum(
        len(list(module.parameters()))
        for module in [sold.dynamics_predictor, sold.reward_predictor, sold.actor, sold.critic, sold.autoencoder]
    )
    assert saved == expected


# BUGS_AND_PITFALLS.md #17 (fixed).
def test_no_autoencoder_graph_without_finetuning(tmp_path, monkeypatch):
    from sold import train_sold

    recorded = []
    original = train_sold.reconstruction_loss

    def recording(autoencoder, images, actions):
        outputs = original(autoencoder, images, actions)
        recorded.append(outputs["reconstruction_loss"].requires_grad)
        return outputs

    monkeypatch.setattr(train_sold, "reconstruction_loss", recording)
    sold = make_tiny_sold(finetune_autoencoder=False, max_steps=15)
    fit_sold(sold, tmp_path)
    assert recorded and not any(recorded)
    # Nor do the dynamics model's image loss gradients accumulate in the frozen weights (under CUDA graphs, such
    # never-reset gradients would point to memory that the next update's graphs overwrite).
    assert all(not p.requires_grad and p.grad is None for p in sold.autoencoder.parameters())
    assert any(p.grad is not None for p in sold.dynamics_predictor.parameters())
