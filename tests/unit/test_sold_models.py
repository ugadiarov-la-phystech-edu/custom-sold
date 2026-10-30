"""modeling/sold: dynamics model, slot aggregation transformer and predictors."""

import pytest
import torch
import torch.distributions as D

from modeling.attention_mask import AlibiMask
from modeling.distributions import TwoHotEncodingDistribution
from modeling.positional_encoding import NoPositionalEncoding
from modeling.sold.dynamics import AutoregressiveWrapper, OCVPSeqDynamicsModel, OCVPSeqLayer, make_ocvp_seq_dynamics_model
from modeling.sold.prediction import GaussianPredictor, Predictor, RMSNorm, ScalarPredictor, TwoHotPredictor
from modeling.sold.slot_aggregation_transformer import SlotAggregationTransformer, TransformerEncoderLayer


# ---------------------------------------------------------------------------------- dynamics
class PlusOnePredictor(torch.nn.Module):
    """Predicts 'slots + 1' at every position and records its input lengths."""

    def __init__(self, input_buffer_size=10):
        super().__init__()
        self.input_buffer_size = input_buffer_size
        self.calls = []

    def forward(self, slots, actions):
        self.calls.append((slots.shape[1], actions.shape[1]))
        return slots + 1


def _slots(batch=2, steps=6, num_slots=2, dim=3):
    return torch.arange(batch * steps * num_slots * dim, dtype=torch.float32).view(batch, steps, num_slots, dim)


def test_autoregressive_free_running_feeds_back_predictions():
    predictor = PlusOnePredictor()
    wrapper = AutoregressiveWrapper(predictor, teacher_forcing=False)
    slots, actions = _slots(), torch.zeros(2, 5, 1)
    predicted = wrapper.predict_slots(slots, actions, steps=3, num_context=2)
    assert predicted.shape == (2, 3, 2, 3)
    for k in range(3):
        assert torch.equal(predicted[:, k], slots[:, 1] + k + 1)
    assert predictor.calls == [(2, 2), (3, 3), (4, 4)]


@pytest.mark.parametrize("batched", [True, False])
def test_autoregressive_teacher_forcing_uses_ground_truth(batched):
    predictor = PlusOnePredictor()
    wrapper = AutoregressiveWrapper(predictor, teacher_forcing=True)
    wrapper.batched_processing = batched
    slots, actions = _slots(), torch.zeros(2, 5, 1)
    predicted = wrapper.predict_slots(slots, actions, steps=3, num_context=2)
    assert torch.equal(predicted, slots[:, 1:4] + 1)
    assert predictor.calls == ([(4, 4)] if batched else [(2, 2), (3, 3), (4, 4)])


def test_autoregressive_wrapper_truncates_to_input_buffer_size():
    predictor = PlusOnePredictor(input_buffer_size=2)
    wrapper = AutoregressiveWrapper(predictor, teacher_forcing=False)
    wrapper.predict_slots(_slots(steps=8), torch.zeros(2, 7, 1), steps=3, num_context=4)
    assert predictor.calls == [(2, 2), (2, 2), (2, 2)]


@pytest.mark.parametrize("residual", [True, False])
def test_ocvp_dynamics_model_shapes(residual):
    model = OCVPSeqDynamicsModel(num_slots=3, slot_dim=5, sequence_length=4, action_dim=2, token_dim=8, hidden_dim=16,
                                 num_layers=2, num_heads=2, residual=residual, input_buffer_size=6).eval()
    slots, actions = torch.randn(2, 4, 3, 5), torch.randn(2, 4, 2)
    out = model(slots, actions)
    assert out.shape == slots.shape
    model.residual = not residual
    other = model(slots, actions)
    assert torch.allclose(out - other, slots if residual else -slots, atol=1e-5)


def test_ocvp_layer_shape():
    layer = OCVPSeqLayer(token_dim=8, hidden_dim=16, n_heads=2).eval()
    assert layer(torch.randn(2, 3, 4, 8)).shape == (2, 3, 4, 8)


def test_make_ocvp_seq_dynamics_model():
    model = make_ocvp_seq_dynamics_model(num_slots=2, slot_dim=4, sequence_length=3, action_dim=2, token_dim=8,
                                         hidden_dim=8, num_layers=1, num_heads=2, input_buffer_size=5,
                                         teacher_forcing=True)
    assert isinstance(model, AutoregressiveWrapper) and model.teacher_forcing
    assert model.input_buffer_size == 5
    out = model.predict_slots(torch.randn(1, 5, 2, 4), torch.randn(1, 4, 2), steps=3, num_context=2)
    assert out.shape == (1, 3, 2, 4)


# ------------------------------------------------------------------ slot aggregation transformer
def _transformer(steps=6, num_slots=2, slot_dim=4, registers=1):
    return SlotAggregationTransformer(AlibiMask(steps, 2, num_slots, registers), NoPositionalEncoding(steps, 8), steps,
                                      num_slots, slot_dim, token_dim=8, num_heads=2, num_layers=2, hidden_dim=16,
                                      num_register_tokens=registers)


def test_slot_aggregation_transformer_output_and_start():
    model = _transformer()
    slots = torch.randn(3, 5, 2, 4)
    assert model(slots).shape == (3, 5, 8)
    assert model(slots, start=4).shape == (3, 1, 8)
    assert torch.allclose(model(slots, start=4)[:, 0], model(slots)[:, -1], atol=1e-5)


def test_slot_aggregation_transformer_is_causal():
    model = _transformer()
    slots = torch.randn(1, 5, 2, 4)
    changed = slots.clone()
    changed[:, 3:] += 1.0
    out, out_changed = model(slots), model(changed)
    assert torch.allclose(out[:, :3], out_changed[:, :3], atol=1e-5)
    assert not torch.allclose(out[:, 3:], out_changed[:, 3:])


def test_transformer_encoder_layer_with_and_without_mask():
    layer = TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16)
    x = torch.randn(2, 5, 8)
    assert layer(x).shape == (2, 5, 8)
    mask = torch.zeros(5, 5)
    assert torch.allclose(layer(x, mask), layer(x), atol=1e-6)


# --------------------------------------------------------------------------------- predictors
PREDICTOR_ARGS = dict(max_episode_steps=6, num_slots=2, slot_dim=4, token_dim=8, num_heads=2, num_layers=1,
                      hidden_dim=16)


def test_rms_norm():
    norm = RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    out = norm(x)
    assert out.dtype == torch.float32  # The float32 weight promotes half-precision inputs.
    expected = x.float() / torch.sqrt((x.float() ** 2).mean() + 1e-6)
    assert torch.allclose(out.float(), expected, atol=1e-2)


@pytest.mark.parametrize("num_mlp_layers", [1, 3])
def test_predictor_mlp_depth(num_mlp_layers):
    model = Predictor(**PREDICTOR_ARGS, output_dim=3, num_mlp_layers=num_mlp_layers)
    linears = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
    assert len(linears) == num_mlp_layers + 1
    assert model(torch.randn(2, 4, 2, 4), start=1).shape == (2, 3, 3)


def test_scalar_predictor_returns_unit_variance_normal():
    dist = ScalarPredictor(**PREDICTOR_ARGS)(torch.randn(2, 3, 2, 4))
    assert isinstance(dist, D.Normal)
    assert dist.mean.shape == (2, 3, 1)
    assert torch.all(dist.stddev == 1.0)


def test_gaussian_predictor_bounds_and_std():
    model = GaussianPredictor(**PREDICTOR_ARGS, output_dim=2, lower_bound=[-0.5, -1.0], upper_bound=[0.5, 1.0])
    with torch.no_grad():
        model.mlp[-1].bias.fill_(100.0)  # Push raw means and stds far out of range.
    dist = model(torch.randn(2, 3, 2, 4))
    assert isinstance(dist, D.Independent)
    assert dist.event_shape == (2,)
    assert torch.allclose(dist.mean, torch.tanh(torch.tensor([0.5, 1.0])).expand(2, 3, 2), atol=1e-6)
    assert torch.all(dist.base_dist.stddev <= 1.0) and torch.all(dist.base_dist.stddev >= 0.1)
    assert dist.rsample().shape == (2, 3, 2)


def test_two_hot_predictor():
    dist = TwoHotPredictor(**PREDICTOR_ARGS)(torch.randn(2, 3, 2, 4), start=2)
    assert isinstance(dist, TwoHotEncodingDistribution)
    assert dist.logits.shape == (2, 1, 255)
    assert dist.mean.shape == (2, 1, 1)
