from modeling.distributions import Moments, TwoHotEncodingDistribution, symexp, symlog
import pytest
import torch
import torch.nn.functional as F


def test_symlog_symexp_are_inverse():
    x = torch.tensor([-100.0, -1.5, 0.0, 0.3, 42.0])
    assert torch.allclose(symexp(symlog(x)), x, rtol=1e-5)
    assert torch.allclose(symlog(x), torch.sign(x) * torch.log1p(x.abs()))


def _dist(logits):
    return TwoHotEncodingDistribution(logits, dims=1)


def test_two_hot_shapes_and_bins():
    logits = torch.randn(4, 3, 255)
    dist = _dist(logits)
    assert dist.bins.shape == (255,)
    assert dist.bins[0] == -20 and dist.bins[-1] == 20
    assert dist._batch_shape == (4, 3)
    assert dist._event_shape == (1,)
    assert dist.mean.shape == (4, 3, 1)
    assert torch.equal(dist.mean, dist.mode)


def test_two_hot_mean_of_one_hot_logits():
    bins = torch.linspace(-20, 20, 255)
    logits = torch.full((1, 255), -1e9)
    logits[0, 200] = 0.0
    assert _dist(logits).mean.item() == pytest.approx(symexp(bins[200]).item(), rel=1e-4)


@pytest.mark.parametrize("value", [-3.0, 0.0, 0.123, 2.5, 1e12, -1e12])
def test_two_hot_log_prob_matches_manual_two_hot_target(value):
    logits = torch.randn(2, 255)
    x = torch.full((2, 1), value)
    dist = _dist(logits)
    y = symlog(x)
    bins = dist.bins
    raw_below = (bins <= y).sum(-1, keepdim=True) - 1
    below, above = raw_below.clamp(min=0), (raw_below + 1).clamp(max=254)
    target = torch.zeros(2, 255)
    for row in range(2):
        b, a = below[row].item(), above[row].item()
        if a == b:
            target[row, b] = 1.0
        else:
            db, da = (y[row] - bins[b]).abs(), (bins[a] - y[row]).abs()
            target[row, b] = da / (da + db)
            target[row, a] = db / (da + db)
    expected = (target * F.log_softmax(logits, -1)).sum(-1)
    assert torch.allclose(dist.log_prob(x), expected, atol=1e-5)


def test_two_hot_log_prob_is_highest_at_the_mode():
    logits = torch.full((1, 255), -10.0)
    logits[0, 127] = 10.0  # Bin 127 is 0.
    dist = _dist(logits)
    assert dist.log_prob(torch.zeros(1, 1)) > dist.log_prob(torch.full((1, 1), 5.0))


def test_moments_tracks_percentiles_with_ema():
    moments = Moments(decay=0.5, max_=1.0, percentile_low=0.0, percentile_high=1.0)
    x = torch.tensor([0.0, 10.0])
    low, invscale = moments(x)
    assert low.item() == pytest.approx(0.0)
    assert moments.high.item() == pytest.approx(5.0)
    assert invscale.item() == pytest.approx(5.0)
    low, invscale = moments(x)
    assert moments.high.item() == pytest.approx(7.5)
    assert invscale.item() == pytest.approx(7.5)
    assert not low.requires_grad and not invscale.requires_grad


def test_moments_invscale_has_a_lower_bound():
    moments = Moments(decay=0.0, max_=2.0)
    _, invscale = moments(torch.zeros(10))
    assert invscale.item() == pytest.approx(0.5)


def test_moments_state_is_saved_in_state_dict():
    moments = Moments()
    moments(torch.arange(100.0))
    state = moments.state_dict()
    assert set(state) == {"low", "high"}


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #33: the two-hot mean is not the expectation of Eq. 5")
def test_two_hot_mean_is_the_expectation_over_symexp_bins():
    logits = torch.full((1, 255), -1e9)
    logits[0, 127] = logits[0, 133] = 0.0  # Half the mass on 0, half on bin 133 (symlog value about 0.94).
    dist = TwoHotEncodingDistribution(logits, dims=1)
    expected = (dist.probs * symexp(dist.bins)).sum(-1, keepdim=True)  # Paper Eq. 5.
    assert torch.allclose(dist.mean, expected, atol=1e-5)
