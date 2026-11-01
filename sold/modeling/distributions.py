from collections.abc import Callable
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# Adapted from: https://github.com/Eclectic-Sheep/sheeprl/blob/c37a39eb4fdc3cb3df92890a39c60545cc451a82/sheeprl/utils/distribution.py#L224
class TwoHotEncodingDistribution:
    def __init__(
        self,
        logits: Tensor,
        dims: int = 0,
        low: int = -20,
        high: int = 20,
        transfwd: Callable[[Tensor], Tensor] = symlog,
        transbwd: Callable[[Tensor], Tensor] = symexp,
    ):
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self._batch_shape = logits.shape[: len(logits.shape) - dims]
        self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

    @property
    def mean(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    @property
    def mode(self) -> Tensor:
        return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

    def log_prob(self, x: Tensor) -> Tensor:
        x = self.transfwd(x)
        # below in [-1, len(self.bins) - 1]
        below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
        # above in [0, len(self.bins)]
        above = below + 1

        # above in [0, len(self.bins) - 1]
        above = torch.minimum(above, torch.full_like(above, len(self.bins) - 1))
        # below in [0, len(self.bins) - 1]
        below = torch.maximum(below, torch.zeros_like(below))

        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        ).squeeze(-2)
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdim=True)
        return (target * log_pred).sum(dim=self.dims)


def quantile(x: Tensor, q: float) -> Tensor:
    """torch.quantile(x, q) with linear interpolation, but traceable by torch.compile.

    torch.quantile checks its input for NaNs, which torch.compile cannot trace; here NaNs are sorted last instead.
    The rank is a Python float (torch.quantile's is float32), which only changes the interpolation weight's last bits.
    """
    values = x.flatten().sort().values
    rank = q * (values.numel() - 1)
    below, above = math.floor(rank), math.ceil(rank)
    return torch.lerp(values[below], values[above], rank - below)


# Adapted from: https://github.com/Eclectic-Sheep/sheeprl/blob/c37a39eb4fdc3cb3df92890a39c60545cc451a82/sheeprl/algos/dreamer_v3/utils.py#L40
class Moments(nn.Module):
    low: Tensor  # Running percentiles (buffers).
    high: Tensor

    def __init__(
        self,
        decay: float = 0.99,
        max_: float = 1,
        percentile_low: float = 0.05,
        percentile_high: float = 0.95,
    ) -> None:
        super().__init__()
        self._decay = decay
        self._max = max_
        self._percentile_low = percentile_low
        self._percentile_high = percentile_high
        self.register_buffer("low", torch.zeros((), dtype=torch.float32))
        self.register_buffer("high", torch.zeros((), dtype=torch.float32))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        low = quantile(x.detach(), self._percentile_low)
        high = quantile(x.detach(), self._percentile_high)
        # In place, so that the buffers keep their memory (needed under CUDA graphs).
        self.low.copy_(self._decay * self.low + (1 - self._decay) * low)
        self.high.copy_(self._decay * self.high + (1 - self._decay) * high)
        invscale = torch.clamp(self.high - self.low, min=1 / self._max)
        return self.low.detach().clone(), invscale.detach()
