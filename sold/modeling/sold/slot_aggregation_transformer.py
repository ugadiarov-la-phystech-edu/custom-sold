from sold.modeling.attention_mask import AttentionMask
from sold.modeling.positional_encoding import PositionalEncoding
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Linear, Module, MultiheadAttention, RMSNorm


class SlotAggregationTransformer(nn.Module):
    def __init__(
        self,
        attention_mask: AttentionMask,
        positional_encoding: PositionalEncoding,
        max_episode_steps: int,
        num_slots: int,
        slot_dim: int,
        token_dim: int,
        num_heads: int,
        num_layers: int,
        hidden_dim: int,
        num_register_tokens: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.num_register_tokens = num_register_tokens
        self.token_dim = token_dim
        self.num_heads = num_heads

        self.transformer_encoder_blocks = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    d_model=self.token_dim,
                    nhead=self.num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # Add cls and register tokens
        self.out_and_register_tokens = nn.Parameter(torch.Tensor(1, 1, 1 + self.num_register_tokens, self.token_dim))
        nn.init.xavier_uniform_(self.out_and_register_tokens)

        self.attention_mask = attention_mask
        self.positional_encoding = positional_encoding
        self.slot_projection = nn.Linear(slot_dim, self.token_dim)

    def forward(self, slots: torch.Tensor, start: int = 0, index: torch.Tensor | None = None) -> torch.Tensor:
        """Output tokens from time step start on, shape (batch_size, sequence_length - start, token_dim).

        With index (a one-element tensor), only the token of that time step, (batch_size, 1, token_dim): the causal mask
        hides the later steps, so slots padded beyond index give the same token as slots ending at index.
        """
        return self._encode(slots, start, return_attention=False, index=index)[0]

    def forward_with_attention(self, slots: torch.Tensor, start: int = 0) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The output tokens and each layer's head-averaged attention weights, (batch_size, num_tokens, num_tokens)."""
        return self._encode(slots, start, return_attention=True)

    def _encode(
        self, slots: torch.Tensor, start: int, return_attention: bool, index: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list]:
        batch_size, sequence_length, num_slots, slot_dim = slots.shape
        slot_tokens = self.slot_projection(slots)  # Project slots to token_dim.

        # Add the output and register tokens to the slots (at every batch and time step).
        out_and_register_tokens = self.out_and_register_tokens.repeat(batch_size, sequence_length, 1, 1)
        tokens = torch.cat((slot_tokens, out_and_register_tokens), dim=2)

        time_encoded_tokens = self.positional_encoding(tokens)

        output_tokens = time_encoded_tokens.reshape(batch_size, -1, self.token_dim)
        attention_mask = self.attention_mask(batch_size, sequence_length)
        attention_weights = []
        for transformer_encoder_block in self.transformer_encoder_blocks:
            output_tokens, weights = transformer_encoder_block.encode(output_tokens, attention_mask, return_attention)
            attention_weights.append(weights)

        output_tokens = output_tokens.reshape(batch_size, sequence_length, -1, self.token_dim)[:, :, -1]
        if index is not None:
            return output_tokens.index_select(1, index), attention_weights
        return output_tokens[:, -(sequence_length - start) :], attention_weights


class TransformerEncoderLayer(Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = nn.SiLU()

    def forward(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        return self.encode(src, src_mask)[0]

    def encode(
        self, src: Tensor, src_mask: Tensor | None = None, return_attention: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        """The layer's output and, with return_attention, its head-averaged attention weights (else None)."""
        x = src
        attention, weights = self._sa_block(self.norm1(x), src_mask, return_attention)
        x = x + attention
        x = x + self._ff_block(self.norm2(x))
        return x, weights

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Tensor | None, need_weights: bool = False
    ) -> tuple[Tensor, Tensor | None]:
        x, weights = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=need_weights)
        return self.dropout1(x), weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
