import torch
import torch.nn as nn

T5_VOCAB_SIZE = 32128  # Rows of T5's embedding matrix; the instruction memories hold raw T5 token ids.


class LanguageEncoder(nn.Module):
    """Encodes an instruction memory (token ids, 0 for empty positions; see sold.envs.wrappers.language) as one
    token_dim vector: the mean of the heard ids' learned embeddings, projected. Independent of word order and of the
    memory length; an empty memory gives the projection's bias."""

    def __init__(self, token_dim: int, vocab_size: int = T5_VOCAB_SIZE, embed_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.projection = nn.Linear(embed_dim, token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: integer ids (..., memory_length). Returns (..., token_dim)."""
        num_heard = (tokens != 0).sum(-1, keepdim=True).clamp(min=1)
        return self.projection(self.embedding(tokens).sum(-2) / num_heard)  # Id 0 embeds to zeros (padding_idx).
