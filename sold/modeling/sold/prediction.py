from .language import LanguageEncoder
from .slot_aggregation_transformer import SlotAggregationTransformer
from collections.abc import Callable, Mapping
from sold.modeling.attention_mask import AlibiMask
from sold.modeling.distributions import TwoHotEncodingDistribution
from sold.modeling.positional_encoding import NoPositionalEncoding
import torch
import torch.distributions as D
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Predictor(nn.Module):
    def __init__(
        self,
        max_episode_steps: int,
        num_slots: int,
        slot_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        num_register_tokens: int = 0,
        num_mlp_layers: int = 1,
        attention_mask: Callable = AlibiMask,
        positional_encoding: Callable = NoPositionalEncoding,
        language: Mapping | None = None,
    ) -> None:
        """Used to predict single quantities like rewards or actions from the set and history of slots.

        attention_mask and positional_encoding are factories (e.g. classes in modeling.attention_mask and
        modeling.positional_encoding) for the slot aggregation transformer. language: the arguments of a
        LanguageEncoder, which conditions the predictions on each step's instruction memory (forward's tokens).
        """
        super().__init__()
        self.language_encoder = None if language is None else LanguageEncoder(token_dim, **language)
        self.slot_aggregation_transformer = SlotAggregationTransformer(
            attention_mask(max_episode_steps, num_heads, num_slots, num_register_tokens),
            positional_encoding(max_episode_steps, token_dim),
            max_episode_steps,
            num_slots,
            slot_dim,
            token_dim,
            num_heads,
            num_layers,
            hidden_dim,
            num_register_tokens,
        )

        self.mlp = []
        for layer_num in range(num_mlp_layers):
            if layer_num == 0:
                self.mlp.append(nn.Linear(token_dim, hidden_dim))
            else:
                self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(RMSNorm(hidden_dim))
            self.mlp.append(nn.SiLU())
        self.mlp.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(
        self,
        slots: torch.Tensor,
        start: int = 0,
        index: torch.Tensor | None = None,
        *,
        tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predictions from time step start on, or only for time step index (see SlotAggregationTransformer).

        tokens: the instruction memory of every step, (batch_size, sequence_length, memory_length), with language.
        """
        features = self.slot_aggregation_transformer(
            slots, start=start, index=index, language=self.encode_language(tokens)
        )
        return self.mlp(features)

    def encode_language(self, tokens: torch.Tensor | None) -> torch.Tensor | None:
        """The encoded instruction memories (batch_size, sequence_length, token_dim), or None without language."""
        if (tokens is None) != (self.language_encoder is None):
            raise ValueError(
                "The predictor was built with language, but got no tokens."
                if tokens is None
                else "The predictor was built without language, but got tokens."
            )
        return None if tokens is None else self.language_encoder(tokens)


class ScalarPredictor(Predictor):
    def __init__(
        self,
        max_episode_steps: int,
        num_slots: int,
        slot_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        num_register_tokens: int = 0,
        num_mlp_layers: int = 1,
        **kwargs,
    ) -> None:
        """Predict scalar outputs like values and rewards directly"""
        super().__init__(
            max_episode_steps,
            num_slots,
            slot_dim,
            token_dim,
            num_heads,
            num_layers,
            hidden_dim,
            output_dim=1,
            num_register_tokens=num_register_tokens,
            num_mlp_layers=num_mlp_layers,
            **kwargs,
        )

    def forward(self, slots: torch.Tensor, start: int = 0, *, tokens: torch.Tensor | None = None) -> D.Distribution:
        # Distributions in float32, also under autocast.
        loc = super().forward(slots, start=start, tokens=tokens).float()
        return D.Normal(loc, 1.0, validate_args=False)


class GaussianPredictor(Predictor):
    def __init__(
        self,
        max_episode_steps: int,
        num_slots: int,
        slot_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        num_register_tokens: int = 0,
        num_mlp_layers: int = 1,
        lower_bound=None,
        upper_bound=None,
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(
            max_episode_steps,
            num_slots,
            slot_dim,
            token_dim,
            num_heads,
            num_layers,
            hidden_dim,
            output_dim=2 * output_dim,
            num_register_tokens=num_register_tokens,
            num_mlp_layers=num_mlp_layers,
            **kwargs,
        )
        self.max_std, self.min_std, self.init_std = max_std, min_std, init_std
        # Non-persistent buffers follow the module's device without changing the checkpoint keys.
        self.register_buffer("lower_bound", torch.as_tensor(lower_bound, dtype=torch.float32), persistent=False)
        self.register_buffer("upper_bound", torch.as_tensor(upper_bound, dtype=torch.float32), persistent=False)

    def forward(
        self,
        slots: torch.Tensor,
        start: int = 0,
        index: torch.Tensor | None = None,
        *,
        tokens: torch.Tensor | None = None,
    ) -> D.Distribution:
        # Distributions in float32, also under autocast.
        x = super().forward(slots, start=start, index=index, tokens=tokens).float()
        mean_, std_ = x.chunk(2, -1)
        mean_ = torch.clamp(mean_, self.lower_bound, self.upper_bound)
        std = (self.max_std - self.min_std) * torch.sigmoid(std_ + self.init_std) + self.min_std
        dist = D.Normal(torch.tanh(mean_), std, validate_args=False)
        return D.Independent(dist, 1, validate_args=False)


class TwoHotPredictor(Predictor):
    def __init__(
        self,
        max_episode_steps: int,
        num_slots: int,
        slot_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        num_register_tokens: int = 0,
        num_mlp_layers: int = 1,
        num_bins: int = 255,
        **kwargs,
    ) -> None:
        """Predict over num_bins exponentially-spaced bins to represent scalar values like rewards."""
        super().__init__(
            max_episode_steps,
            num_slots,
            slot_dim,
            token_dim,
            num_heads,
            num_layers,
            hidden_dim,
            output_dim=num_bins,
            num_register_tokens=num_register_tokens,
            num_mlp_layers=num_mlp_layers,
            **kwargs,
        )

    def forward(
        self, slots: torch.Tensor, start: int = 0, *, tokens: torch.Tensor | None = None
    ) -> TwoHotEncodingDistribution:
        logits = super().forward(slots, start=start, tokens=tokens)
        return TwoHotEncodingDistribution(logits, dims=1)
