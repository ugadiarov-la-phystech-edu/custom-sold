"""train_sold.SOLDModule: construction, losses, imagination and action selection."""

import pytest
import torch

from modeling.distributions import TwoHotEncodingDistribution
from tests import fakes
from tests.conftest import make_tiny_sold
from train_sold import SOLDModule


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
    assert tiny_sold.sequence_length == 5 and tiny_sold.automatic_optimization is False
    assert torch.allclose(tiny_sold.discounts, 0.96 ** torch.arange(3.0).unsqueeze(0))
    assert "env" not in tiny_sold.hparams and tiny_sold.hparams.imagination_horizon == 3
    assert tiny_sold.critic_target is not tiny_sold.critic
    for a, b in zip(tiny_sold.critic.parameters(), tiny_sold.critic_target.parameters()):
        assert torch.equal(a, b)
    assert tiny_sold.dynamics_predictor.input_buffer_size == 5
    assert torch.equal(tiny_sold.actor.lower_bound, torch.tensor([-1.0, -1.0]))


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


def _reference_lambda_returns(rewards, values, gamma, lam):
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1]
    for t in reversed(range(horizon)):
        next_return = rewards[:, t] + gamma * (1 - lam) * values[:, t] + gamma * lam * next_return
        returns[:, t] = next_return
    return returns


def test_compute_lambda_returns(tiny_sold):
    rewards, values = torch.randn(4, 3), torch.randn(4, 3)
    expected = _reference_lambda_returns(rewards, values, 0.96, 0.95)
    assert torch.allclose(tiny_sold.compute_lambda_returns(rewards, values), expected, atol=1e-6)


@pytest.mark.parametrize("gradients", ["dynamics", "reinforce"])
def test_compute_actor_loss(gradients):
    sold = make_tiny_sold(actor_gradients=gradients)
    returns = torch.randn(2, 3, requires_grad=True)
    values = torch.randn(2, 3)
    log_probs = torch.randn(2, 3, requires_grad=True)
    entropies = torch.rand(2, 3)
    losses = sold.compute_actor_loss(returns, values, log_probs, entropies)
    assert set(losses) == {"actor_loss", "actor_return_loss", "actor_entropy_loss"}
    assert torch.allclose(losses["actor_loss"], losses["actor_return_loss"] + losses["actor_entropy_loss"])
    expected_entropy = -3e-4 * torch.mean(sold.discounts * entropies)
    assert torch.allclose(losses["actor_entropy_loss"], expected_entropy)
    losses["actor_loss"].backward()
    if gradients == "dynamics":
        assert returns.grad is not None and log_probs.grad is None
    else:
        assert log_probs.grad is not None and returns.grad is None


def test_compute_actor_loss_rejects_unknown_gradients(tiny_sold):
    tiny_sold.actor_gradients = "magic"
    with pytest.raises(ValueError, match="Invalid actor_gradients"):
        tiny_sold.compute_actor_loss(torch.randn(2, 3), torch.randn(2, 3), torch.randn(2, 3), torch.rand(2, 3))


def test_compute_critic_loss(tiny_sold):
    logits = torch.randn(2, 3, 255, requires_grad=True)
    dist = TwoHotEncodingDistribution(logits, dims=1)
    returns, targets = torch.randn(2, 3), torch.randn(2, 3)
    losses = tiny_sold.compute_critic_loss(dist, returns, targets)
    expected_return = torch.mean(tiny_sold.discounts * -dist.log_prob(returns.unsqueeze(2)))
    assert torch.allclose(losses["critic_return_loss"], expected_return)
    assert torch.allclose(losses["critic_loss"], losses["critic_return_loss"] + losses["critic_target_regularization_loss"])
    assert isinstance(losses["return_mse_loss"], float)


def test_compute_dynamics_loss(tiny_sold):
    images, actions, _ = _batch(tiny_sold)
    slots = tiny_sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    tiny_sold.after_eval = False
    losses = tiny_sold.compute_dynamics_loss(images, slots, actions)
    assert torch.allclose(losses["dynamics_loss"], losses["slot_loss"] + losses["image_loss"])
    assert tiny_sold.dynamics_predictor.training


def test_compute_reward_loss_ignores_first_step(tiny_sold):
    images, actions, rewards = _batch(tiny_sold)
    outputs = tiny_sold.autoencoder(images, actions[:, 1:])
    tiny_sold.after_eval = False
    losses = tiny_sold.compute_reward_loss(images, outputs["reconstructions"], outputs["slots"], rewards)
    assert torch.isfinite(losses["reward_loss"]) and isinstance(losses["reward_mse_loss"], float)


@pytest.mark.parametrize("from_every", [False, True])
def test_imagine_ahead_shapes(from_every):
    sold = make_tiny_sold(start_imagination_from_every=from_every)
    sold.after_eval = False
    images, actions, _ = _batch(sold)
    slots = sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    returns, values_targ, values_dist, log_probs, entropies = sold.imagine_ahead(slots, actions)
    batch = 2 * (5 - 2 + 1) if from_every else 2
    assert returns.shape == values_targ.shape == log_probs.shape == (batch, 3)
    assert entropies.shape == (batch, 3, 1)  # Broadcasts against discounts (1, 3) in the actor loss (#26).
    assert values_dist.mean.shape == (batch, 3, 1)
    assert not sold.dynamics_predictor.training
    assert all(p.requires_grad for p in sold.critic.parameters())  # Unfrozen again.
    returns.mean().backward()
    assert any(p.grad is not None for p in sold.actor.parameters())
    assert all(p.grad is None for p in sold.critic.parameters())


def test_visualizations_after_evaluation(tiny_sold, monkeypatch):
    recorder = LogRecorder()
    monkeypatch.setattr(tiny_sold, "log", recorder)
    tiny_sold.after_eval = True
    images, actions, rewards = _batch(tiny_sold)
    outputs = tiny_sold.autoencoder(images, actions[:, 1:])
    slots = outputs["slots"].detach()
    tiny_sold.compute_dynamics_loss(images, slots, actions)
    tiny_sold.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
    tiny_sold.imagine_ahead(slots, actions)
    assert recorder.names == ["dynamics_prediction", "reward_prediction", "reward_predictor_attention",
                              "latent_imagination", "actor_attention"]


def test_visualizations_without_slot_masks(monkeypatch):
    import functools
    from modeling.autoencoder.cnn import CnnDecoder, CnnEncoder
    from modeling.autoencoder.cnn.autoencoder import Cnn
    autoencoder = Cnn(CnnEncoder([4, 4], [3, 3], [2, 2]),
                      functools.partial(CnnDecoder, image_size=(16, 16), num_channels=[4, 4], kernel_sizes=[3, 3],
                                        strides=[2, 2]))
    sold = make_tiny_sold(autoencoder=autoencoder)
    recorder = LogRecorder()
    monkeypatch.setattr(sold, "log", recorder)
    sold.after_eval = True
    images, actions, rewards = _batch(sold)
    outputs = sold.autoencoder(images, actions[:, 1:])
    sold.compute_reward_loss(images, outputs["reconstructions"], outputs["slots"], rewards)
    sold.imagine_ahead(outputs["slots"], actions)
    assert recorder.names == ["reward_prediction", "latent_imagination"]  # No attention images without masks.


def test_select_action_modes(tiny_sold):
    obs = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
    random_action = tiny_sold.select_action(obs, is_first=True, mode="random")
    assert random_action.shape == (2,) and tiny_sold._slot_history.shape == (1, 1, 2, 8)
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
        low=-0.1, high=0.1, shape=(2,), dtype=env.action_space.dtype)
    sold = make_tiny_sold(env)
    obs = torch.zeros(3, 16, 16, dtype=torch.uint8)
    for _ in range(5):
        assert sold.select_action(obs, is_first=True, mode="train").abs().max() <= 0.1
