import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import Any, Dict


def actor_loss(lambda_returns: torch.Tensor, predicted_values_targ: torch.Tensor, action_log_probs: torch.Tensor,
               action_entropies: torch.Tensor, discounts: torch.Tensor, offset: torch.Tensor, invscale: torch.Tensor,
               gradients: str, entropy_weight: float) -> Dict[str, Any]:
    """Actor loss on normalized advantages, with an entropy bonus.

    lambda_returns, predicted_values_targ (target critic) and action_log_probs: (batch_size, horizon); action_entropies:
    (batch_size, horizon, 1), broadcast against discounts (1, horizon) (BUGS_AND_PITFALLS.md #26). Returns and values
    are normalized as (x - offset) / invscale. gradients: "dynamics" (through the returns) or "reinforce".
    """
    normed_lambda_returns = (lambda_returns[:, :-1] - offset) / invscale
    normed_base = (predicted_values_targ[:, :-1] - offset) / invscale
    advantage = normed_lambda_returns - normed_base

    if gradients == "dynamics":
        return_loss = -torch.mean(discounts.detach()[:, :-1] * advantage)
    elif gradients == "reinforce":
        return_loss = torch.mean(action_log_probs[:, :-1] * advantage.detach())
    else:
        raise ValueError(f"Invalid actor_gradients: {gradients}.")

    entropy_loss = -torch.mean(discounts.detach() * action_entropies)
    return {"actor_loss": return_loss + entropy_weight * entropy_loss, "actor_return_loss": return_loss,
            "actor_entropy_loss": entropy_weight * entropy_loss}


def critic_loss(predicted_values_dist: D.Distribution, lambda_returns: torch.Tensor,
                predicted_values_targ: torch.Tensor, discounts: torch.Tensor,
                regularization_weight: float) -> Dict[str, Any]:
    """Negative log-likelihood of the λ-returns, regularized towards the target critic's values.

    predicted_values_dist's mean: (batch_size, horizon, 1); lambda_returns and predicted_values_targ: (batch_size,
    horizon); discounts: (1, horizon).
    """
    return_loss = torch.mean(
        discounts.detach() * (-predicted_values_dist.log_prob(lambda_returns.detach().unsqueeze(2))))
    target_regularization_loss = torch.mean(
        discounts.detach() * (-predicted_values_dist.log_prob(predicted_values_targ.detach().unsqueeze(2))))
    return {"critic_loss": return_loss + regularization_weight * target_regularization_loss,
            "critic_return_loss": return_loss,
            "critic_target_regularization_loss": regularization_weight * target_regularization_loss,
            "return_mse_loss": F.mse_loss(predicted_values_dist.mean.squeeze(2), lambda_returns).item()}
