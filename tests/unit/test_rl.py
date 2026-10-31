"""rl.returns and rl.losses: SOLD's behavior-learning math as pure functions."""

import pytest
from sold.modeling.distributions import TwoHotEncodingDistribution
from sold.rl.losses import actor_loss, critic_loss
from sold.rl.returns import ema_update, lambda_returns
import torch

NO_NORMALIZATION = (0.0, 1.0)  # offset, invscale


# ------------------------------------------------------------------------------------ returns
def test_lambda_returns_by_hand():
    # γ = λ = 0.5: R_2 = v_1 = 4, R_1 = 2 + 0.25 * 4 + 0.25 * 4 = 4, R_0 = 1 + 0.25 * 3 + 0.25 * 4 = 2.75.
    returns = lambda_returns(torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]]), 0.5, 0.5)
    assert torch.allclose(returns, torch.tensor([[2.75, 4.0]]))


def _reference_lambda_returns(rewards, values, gamma, lam):
    """Independent formulation: R_t = r_t + γ(1 - λ) v_t + γλ R_{t+1}, R_horizon = v_last."""
    returns = torch.zeros_like(rewards)
    next_return = values[:, -1]
    for t in reversed(range(rewards.shape[1])):
        next_return = rewards[:, t] + gamma * (1 - lam) * values[:, t] + gamma * lam * next_return
        returns[:, t] = next_return
    return returns


def test_lambda_returns_match_an_independent_formulation():
    rewards, values = torch.randn(4, 3), torch.randn(4, 3)
    expected = _reference_lambda_returns(rewards, values, 0.96, 0.95)
    assert torch.allclose(lambda_returns(rewards, values, 0.96, 0.95), expected, atol=1e-6)


def test_lambda_returns_limits():
    rewards, values = torch.randn(3, 4), torch.randn(3, 4)
    # λ = 0: one-step targets r_t + γ v_t.
    assert torch.allclose(lambda_returns(rewards, values, 0.9, 0.0), rewards + 0.9 * values)
    # γ = 0: only the immediate reward counts.
    assert torch.allclose(lambda_returns(rewards, values, 0.0, 0.95), rewards)


def test_ema_update():
    target, source = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    expected = [0.75 * t.detach() + 0.25 * s.detach() for t, s in zip(target.parameters(), source.parameters())]
    ema_update(target, source, decay=0.75)
    assert all(torch.allclose(t, e) for t, e in zip(target.parameters(), expected))


# ------------------------------------------------------------------------------------- losses
def test_actor_loss_normalizes_returns_and_values():
    returns, values = torch.tensor([[5.0, 3.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0]])
    losses = actor_loss(
        returns, values, torch.zeros(1, 3), torch.zeros(1, 3), torch.ones(1, 3), 1.0, 2.0, "dynamics", 0.0
    )
    assert torch.isclose(losses["actor_return_loss"], torch.tensor(-((5.0 - 1.0) / 2 + (3.0 - 1.0) / 2) / 2))


def test_actor_loss_dynamics_gradients_by_hand():
    returns = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    values = torch.tensor([[1.0, 1.0, 0.0]])
    entropies = torch.tensor([[1.0, 1.0, 1.0]])
    discounts = torch.tensor([[1.0, 0.5, 0.25]])
    losses = actor_loss(returns, values, torch.zeros(1, 3), entropies, discounts, *NO_NORMALIZATION, "dynamics", 0.1)
    # The last step only bootstraps: advantages (2, 1) weighted by discounts (1, 0.5).
    assert torch.isclose(losses["actor_return_loss"], torch.tensor(-(2.0 * 1.0 + 1.0 * 0.5) / 2))
    assert torch.isclose(losses["actor_entropy_loss"], torch.tensor(-0.1 * (1.0 + 0.5 + 0.25) / 3))
    assert torch.isclose(losses["actor_loss"], losses["actor_return_loss"] + losses["actor_entropy_loss"])
    losses["actor_loss"].backward()
    assert returns.grad is not None


def test_actor_loss_reinforce_gradients_go_through_log_probs_only():
    returns = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
    log_probs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
    losses = actor_loss(
        returns, torch.zeros(1, 3), log_probs, torch.zeros(1, 3), torch.ones(1, 3), *NO_NORMALIZATION, "reinforce", 0.0
    )
    assert torch.isclose(losses["actor_return_loss"], torch.tensor((-1.0 * 3.0 - 2.0 * 2.0) / 2))
    losses["actor_loss"].backward()
    assert log_probs.grad is not None and returns.grad is None


def test_actor_loss_rejects_unknown_gradients():
    with pytest.raises(ValueError, match="Invalid actor_gradients: magic"):
        actor_loss(
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.zeros(1, 3),
            torch.ones(1, 3),
            *NO_NORMALIZATION,
            "magic",
            0.0,
        )


def test_critic_loss():
    dist = TwoHotEncodingDistribution(torch.randn(2, 3, 255, requires_grad=True), dims=1)
    returns, targets, discounts = torch.randn(2, 3), torch.randn(2, 3), torch.tensor([[1.0, 0.5, 0.25]])
    losses = critic_loss(dist, returns, targets, discounts, regularization_weight=0.5)
    expected_return = torch.mean(discounts * -dist.log_prob(returns.unsqueeze(2)))
    expected_regularization = torch.mean(discounts * -dist.log_prob(targets.unsqueeze(2)))
    assert torch.allclose(losses["critic_return_loss"], expected_return)
    assert torch.allclose(losses["critic_target_regularization_loss"], 0.5 * expected_regularization)
    assert torch.allclose(losses["critic_loss"], expected_return + 0.5 * expected_regularization)
    assert torch.allclose(losses["return_mse_loss"], torch.mean((dist.mean.squeeze(2) - returns) ** 2))
    assert not losses["return_mse_loss"].requires_grad
