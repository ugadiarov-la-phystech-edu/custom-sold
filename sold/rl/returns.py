import torch
import torch.nn as nn


def lambda_returns(
    rewards: torch.Tensor, values: torch.Tensor, discount_factor: float, return_lambda: float
) -> torch.Tensor:
    """λ-returns of imagined trajectories, bootstrapped with the last step's value.

    R_t = r_t + γ(1 - λ) v_t + γλ R_{t+1}, R_horizon = v_{horizon - 1}. rewards, values and the result have shape
    (batch_size, horizon).
    """
    returns = [values[:, -1:]]
    interm = rewards + discount_factor * values * (1 - return_lambda)
    for t in reversed(range(rewards.shape[1])):
        returns.append(interm[:, t].unsqueeze(1) + discount_factor * return_lambda * returns[-1])
    return torch.cat(list(reversed(returns)), dim=1)[:, :-1]


def ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """target = decay * target + (1 - decay) * source, in place."""
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_((1 - decay) * source_param.data + decay * target_param.data)
