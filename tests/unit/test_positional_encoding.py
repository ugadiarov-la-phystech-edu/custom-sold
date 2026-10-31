import math
import pytest
from sold.modeling.positional_encoding import (
    LearnedEncoding,
    NoPositionalEncoding,
    SinusoidalEncoding,
    SinusoidalPositionalEncoding,
    TokenWiseSinusoidalPositionalEncoding,
)
import torch


def _sinusoid(max_len, dim):
    table = torch.zeros(max_len, dim)
    for pos in range(max_len):
        for i in range(0, dim, 2):
            angle = pos / (10000 ** (i / dim))
            table[pos, i] = math.sin(angle)
            table[pos, i + 1] = math.cos(angle)
    return table


def test_no_positional_encoding_is_identity():
    tokens = torch.randn(2, 3, 4, 6)
    assert torch.equal(NoPositionalEncoding(5, 6)(tokens), tokens)


def test_learned_encoding_is_shared_across_slots():
    encoding = LearnedEncoding(5, 6)
    assert isinstance(encoding.positional_encoding, torch.nn.Parameter)
    tokens = torch.zeros(2, 3, 4, 6)
    out = encoding(tokens)
    assert torch.equal(out[0, :, 0], encoding.positional_encoding[:3].detach())
    assert torch.equal(out[1, :, 3], out[0, :, 0])


def test_sinusoidal_encoding_values():
    encoding = SinusoidalEncoding(5, 6)
    assert torch.allclose(encoding.positional_encoding, _sinusoid(5, 6), atol=1e-5)
    out = encoding(torch.zeros(1, 4, 2, 6))
    assert torch.allclose(out[0, :, 1], _sinusoid(5, 6)[:4], atol=1e-5)


@pytest.mark.parametrize("shape, match", [((1, 6, 2, 6), "sequence length"), ((1, 2, 2, 4), "token_dim")])
def test_positional_encoding_input_validation(shape, match):
    with pytest.raises(ValueError, match=match):
        NoPositionalEncoding(5, 6)(torch.zeros(shape))


def test_positional_encoding_table_validation():
    encoding = NoPositionalEncoding(5, 6)
    encoding.positional_encoding = torch.zeros(4, 6)
    with pytest.raises(ValueError, match="positional encoding of shape"):
        encoding(torch.zeros(1, 2, 2, 6))


def test_sinusoidal_positional_encoding_per_time_step():
    encoding = SinusoidalPositionalEncoding(d_model=6, dropout=0.0, max_len=5)
    x = torch.zeros(2, 3, 4, 6)
    out = encoding(x, batch_size=2, num_slots=4)
    assert out.shape == (2, 3, 4, 6)
    assert torch.allclose(out[1, :, 2], _sinusoid(5, 6)[:3], atol=1e-5)


def test_sinusoidal_positional_encoding_follows_input_device():
    # 'pe' is a plain attribute, not a buffer, so it is moved lazily to the input's device.
    encoding = SinusoidalPositionalEncoding(d_model=4, dropout=0.0, max_len=3)
    out = encoding(torch.zeros(1, 2, 1, 4, device="meta"), batch_size=1, num_slots=1)
    assert encoding.pe.device.type == "meta"
    assert out.shape == (1, 2, 1, 4)
    assert "pe" not in encoding.state_dict()


def test_token_wise_encoding_pads_incomplete_time_steps():
    encoding = TokenWiseSinusoidalPositionalEncoding(d_model=4, dropout=0.0, max_len=3)
    table = _sinusoid(3, 4)
    out = encoding(torch.zeros(1, 5, 4), batch_size=1, num_slots=2)  # 2.5 time steps of 2 tokens.
    assert out.shape == (1, 5, 4)
    assert torch.allclose(out[0], table.repeat_interleave(2, dim=0)[:5], atol=1e-5)
    full = encoding(torch.zeros(1, 4, 4), batch_size=1, num_slots=2)
    assert torch.allclose(full[0], table.repeat_interleave(2, dim=0)[:4], atol=1e-5)
