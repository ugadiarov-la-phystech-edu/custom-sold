import math
import matplotlib.pyplot as plt
from modeling.attention_mask import AlibiMask, CausalMask, fill_with_neg_inf
import pytest
import torch

NEG_INF = float("-inf")


def test_fill_with_neg_inf_keeps_dtype():
    out = fill_with_neg_inf(torch.zeros(2, 3, dtype=torch.float16))
    assert out.dtype == torch.float16
    assert torch.isneginf(out).all()


def test_causal_mask_blocks_future_time_steps():
    mask = CausalMask(max_sequence_length=3, num_heads=2, num_slots=2, num_register_tokens=1)
    assert mask.step_size == 4 and mask.max_num_tokens == 12
    full = mask._full_mask
    assert full.shape == (2, 12, 12)
    for i in range(12):
        for j in range(12):
            expected = 0.0 if j // 4 <= i // 4 else NEG_INF
            assert full[0, i, j].item() == expected
    assert torch.equal(full[0], full[1])


def test_forward_returns_last_window_repeated_per_batch():
    mask = CausalMask(max_sequence_length=4, num_heads=2, num_slots=1)
    out = mask(batch_size=3, sequence_length=2)
    assert out.shape == (3 * 2, 4, 4)
    assert torch.equal(out[0], mask._full_mask[0, -4:, -4:])
    assert torch.equal(out[5], mask._full_mask[1, -4:, -4:])


def test_forward_rejects_too_long_sequences():
    with pytest.raises(ValueError, match="sequence length"):
        CausalMask(2, 1, 1)(batch_size=1, sequence_length=3)


def test_forward_rejects_a_mask_of_the_wrong_size():
    mask = CausalMask(2, 1, 1)
    mask._full_mask = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError, match="mask of size"):
        mask(batch_size=1, sequence_length=1)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_alibi_slopes_power_of_two(num_heads):
    slopes = AlibiMask(2, num_heads, 1).get_slopes(num_heads)
    start = 2 ** (-(2 ** -(math.log2(num_heads) - 2)))
    assert slopes == pytest.approx([start ** (i + 1) for i in range(num_heads)])


def test_alibi_slopes_not_power_of_two():
    mask = AlibiMask(2, 6, 1)
    assert mask.get_slopes(6) == pytest.approx(mask.get_slopes(4) + mask.get_slopes(8)[0::2][:2])
    assert len(mask.get_slopes(6)) == 6


def test_alibi_mask_values():
    num_slots, registers, steps, heads = 2, 1, 3, 2
    mask = AlibiMask(steps, heads, num_slots, registers)
    full = mask._full_mask
    step = num_slots + 1 + registers
    slopes = mask.get_slopes(heads)
    assert full.shape == (heads, steps * step, steps * step)
    for h in range(heads):
        for i in range(steps * step):
            for j in range(steps * step):
                ti, tj, token_j = i // step, j // step, j % step
                if tj > ti or (tj < ti and token_j >= num_slots):
                    assert full[h, i, j].item() == NEG_INF
                else:
                    assert full[h, i, j].item() == pytest.approx(slopes[h] * tj)


def test_show_draws_the_mask(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gcf()))
    CausalMask(2, 2, 1).show(head=1)
    assert len(shown) == 1
    assert "CausalMask" in shown[0].axes[0].get_title()
    plt.close("all")


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #32: ALiBi slopes differ from the original formula")
def test_alibi_slopes_match_press_et_al():
    assert AlibiMask(2, 8, 1).get_slopes(8) == pytest.approx([2 ** -(i + 1) for i in range(8)])


# BUGS_AND_PITFALLS.md #20 (fixed).
def test_wrong_mask_size_raises_value_error():
    mask = CausalMask(2, 1, 1)
    mask._full_mask = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError):
        mask(batch_size=1, sequence_length=1)


# BUGS_AND_PITFALLS.md #21 (fixed).
def test_mask_plot_title_shows_the_requested_head(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    CausalMask(2, 2, 1).show(head=1)
    title = plt.gca().get_title()
    plt.close("all")
    assert title.endswith("(head=1)")
