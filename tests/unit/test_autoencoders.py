"""modeling/autoencoder: base class, CNN autoencoder, SAVi and its parts, logging callbacks."""

import functools
import hydra
from modeling.autoencoder.base import Autoencoder
from modeling.autoencoder.cnn import CnnDecoder, CnnEncoder
from modeling.autoencoder.cnn.autoencoder import Cnn, get_embedding_dim
from modeling.autoencoder.cnn.decoder import Decoder
from modeling.autoencoder.cnn.encoder import SamePadConv2d
from modeling.autoencoder.logging import LoggingCallback, LogReconstruction
from modeling.autoencoder.savi import Corrector, SaviCnnDecoder, SaviCnnEncoder
from modeling.autoencoder.savi.encoder import ImagePositionalEmbedding
from modeling.autoencoder.savi.initializer import Learned, LearnedRandom, SlotInitializer
from modeling.autoencoder.savi.predictor import (
    ActionConditionalTransformerPredictor,
    MetaAttention,
    MultiHeadSelfAttention,
    TransformerBlock,
    TransformerPredictor,
    init_xavier_,
)
from omegaconf import OmegaConf
import os
import pytest
from tests.conftest import CONFIGS_DIR, make_tiny_savi
import torch
from unittest import mock


# ----------------------------------------------------------------------------------- CNN parts
def test_same_pad_conv_keeps_size_divided_by_stride():
    conv = SamePadConv2d(3, 4, kernel_size=4, stride=2)
    assert SamePadConv2d.get_same_padding(15, 4, 2, 1) == 3
    assert conv(torch.randn(1, 3, 15, 15)).shape == (1, 4, 8, 8)
    no_pad = SamePadConv2d(3, 4, kernel_size=1, stride=1)
    assert SamePadConv2d.get_same_padding(9, 1, 1, 1) == 0
    assert no_pad(torch.randn(1, 3, 9, 9)).shape == (1, 4, 9, 9)


def test_cnn_encoder_validation():
    with pytest.raises(ValueError, match="same length"):
        CnnEncoder([4, 4], [3], [1, 1])
    encoder = CnnEncoder([4, 6], [3, 3], [2, 1], in_channels=3)
    assert isinstance(encoder.encoder[-1], torch.nn.Identity)
    assert encoder(torch.randn(2, 3, 8, 8)).shape == (2, 6, 4, 4)
    with pytest.raises(ValueError, match="in_channels"):
        encoder(torch.randn(2, 1, 8, 8))


def test_cnn_decoder():
    with pytest.raises(AssertionError):
        CnnDecoder(10, (8, 8), [4], [3, 3], [2, 2])
    decoder = CnnDecoder(embedding_dim=10, image_size=(8, 8), num_channels=[6, 4], kernel_sizes=[4, 4], strides=[2, 2])
    assert decoder.feature_map_size.tolist() == [2, 2]
    assert (decoder.width, decoder.height) == (8, 8)
    assert CnnDecoder.get_same_padding(4, 2, 1) == (1, 0)
    assert CnnDecoder.get_same_padding(3, 2, 1) == (1, 1)
    out = decoder(torch.randn(5, 10))
    assert out.shape == (5, 1, 3, 8, 8)
    assert torch.all((out >= 0) & (out <= 1))
    assert isinstance(decoder.decoder[-1], torch.nn.Sigmoid)


def test_decoder_base_is_abstract():
    with pytest.raises(TypeError):
        Decoder((8, 8), 3)
    assert Decoder.forward(None, None) is None


def _cnn_autoencoder():
    encoder = CnnEncoder([4, 6], [3, 3], [2, 2])
    decoder = functools.partial(CnnDecoder, image_size=(8, 8), num_channels=[6, 4], kernel_sizes=[3, 3], strides=[2, 2])
    return Cnn(encoder, decoder)


def test_cnn_autoencoder_round_trip():
    model = _cnn_autoencoder()
    assert get_embedding_dim(model.encoder, (8, 8)) == 6 * 2 * 2
    assert (model.num_slots, model.slot_dim) == (1, 24)
    images = torch.rand(2, 3, 3, 8, 8)
    outputs = model(images, actions=torch.zeros(2, 2, 1))
    assert outputs["slots"].shape == (2, 3, 1, 24)
    assert outputs["reconstructions"].shape == images.shape


def test_base_visualize_reconstruction_with_and_without_images():
    model = _cnn_autoencoder()
    images = torch.rand(4, 3, 8, 8)
    both = model.visualize_reconstruction({"images": images, "reconstructions": images})
    only = model.visualize_reconstruction({"reconstructions": images})
    assert both.shape[0] == 3 and only.shape[0] == 3
    assert both.shape[1] > only.shape[1]
    assert both.shape[2] == only.shape[2] == 4 * 10 + 2


def test_base_forward_requires_reconstructions():
    class NoReconstructions(Autoencoder):
        num_slots = 1
        slot_dim = 2

        def encode(self, images, actions, prior_slots=None):
            return torch.zeros(images.shape[0], images.shape[1], 1, 2)

        def decode(self, slots):
            return {"rgbs": slots}

    with pytest.raises(ValueError, match="reconstructions"):
        NoReconstructions()(torch.zeros(1, 2, 3, 4, 4), torch.zeros(1, 1, 1))
    # The abstract property bodies only 'pass'.
    assert Autoencoder.num_slots.fget(None) is None and Autoencoder.slot_dim.fget(None) is None
    assert Autoencoder.encode(None, None, None) is None and Autoencoder.decode(None, None) is None


# BUGS_AND_PITFALLS.md #23 (fixed).
def test_cnn_autoencoder_config_can_be_instantiated():
    from utils.instantiate import fill_in_missing

    cfg = OmegaConf.load(CONFIGS_DIR / "autoencoder" / "cnn.yaml")
    fill_in_missing(cfg, {"image_size": [16, 16]})  # As train_autoencoder.py does with the dataset infos.
    autoencoder = hydra.utils.instantiate(cfg)
    assert autoencoder(torch.rand(1, 2, 3, 16, 16), torch.zeros(1, 1, 2))["reconstructions"].shape == (1, 2, 3, 16, 16)


# ------------------------------------------------------------------------------------- SAVi
def test_image_positional_embedding_grid():
    embedding = ImagePositionalEmbedding(feature_dim=5, image_size=(3, 4))
    grid = embedding.grid[0]
    assert grid.shape == (4, 3, 4)
    assert torch.allclose(grid[0, :, 0], torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.allclose(grid[1, 0], torch.linspace(-1, 1, 4))
    assert torch.allclose(grid[2:], 1 - grid[:2])
    assert embedding(torch.zeros(2, 3, 4, 5)).shape == (2, 3, 4, 5)


def test_savi_encoder_and_decoder_shapes():
    encoder = SaviCnnEncoder((8, 8), [4, 4], [3, 3], [1, 1], feature_dim=6)
    assert encoder(torch.rand(2, 3, 8, 8)).shape == (2, 64, 6)
    decoder = SaviCnnDecoder((8, 8), [4], [3], [1], in_channels=5)
    rgbs, masks = decoder(torch.randn(2, 3, 5))
    assert rgbs.shape == (2, 3, 3, 8, 8) and masks.shape == (2, 3, 1, 8, 8)
    assert torch.allclose(masks.sum(1), torch.ones(2, 1, 8, 8))
    assert rgbs.min() >= 0 and rgbs.max() <= 1
    with pytest.raises(ValueError, match="slot dimension"):
        decoder(torch.randn(2, 3, 4))


def test_corrector_iterations_and_attention():
    corrector = Corrector(num_slots=3, slot_dim=4, feature_dim=6, num_iterations=1, num_initial_iterations=3)
    features, slots = torch.randn(2, 10, 6), torch.randn(2, 3, 4)
    gru_calls = []
    corrector.gru.register_forward_hook(lambda *args: gru_calls.append(1))
    out = corrector(features, slots, is_first=True)
    assert len(gru_calls) == 3
    corrector(features, slots, is_first=False)
    assert len(gru_calls) == 4
    assert out.shape == (2, 3, 4)
    masks = corrector.get_attention_masks()
    assert masks.shape == (2, 3, 10)
    assert torch.allclose(masks.sum(-1), torch.ones(2, 3))


def test_slot_initializers():
    learned = Learned(3, 4)
    assert learned(2).shape == (2, 3, 4)
    assert torch.equal(learned(2)[0], learned(2)[1])
    random_init = LearnedRandom(3, 4)
    first, second = random_init(2), random_init(2)
    assert first.shape == (2, 3, 4) and not torch.equal(first, second)
    with pytest.raises(TypeError):
        SlotInitializer(3, 4)
    assert SlotInitializer.forward(learned, 2) is None


def test_init_xavier_zeroes_biases_and_skips_vectors():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    with torch.no_grad():
        model[0].bias.fill_(1.0)
        model[1].weight.fill_(3.0)
    init_xavier_(model)
    assert torch.all(model[0].bias == 0)
    assert torch.all(model[1].weight == 3.0)


def test_meta_attention_validation_and_masks():
    with pytest.raises(AssertionError):
        MetaAttention(emb_dim=6, num_heads=4)
    attention = MultiHeadSelfAttention(emb_dim=8, num_heads=2)
    with pytest.raises(AssertionError):
        attention.get_attention_masks()
    out = attention(torch.randn(3, 5, 8))
    assert out.shape == (3, 5, 8)
    assert attention.get_attention_masks().shape == (3 * 2, 5, 5)


@pytest.mark.parametrize("pre_norm", [True, False])
def test_transformer_block(pre_norm):
    block = TransformerBlock(embed_dim=8, num_heads=2, mlp_size=16, pre_norm=pre_norm)
    assert block(torch.randn(2, 3, 8)).shape == (2, 3, 8)
    with pytest.raises(AssertionError):
        block(torch.randn(2, 8))


def test_transformer_predictors():
    slots = torch.randn(2, 3, 8)
    plain = TransformerPredictor(slot_dim=8, action_dim=2, num_heads=2, mlp_size=16)
    assert torch.equal(plain(slots, torch.zeros(2, 2)), plain(slots, torch.ones(2, 2)))
    from modeling.autoencoder.savi.predictor import Predictor as SaviPredictor

    assert SaviPredictor.forward(None, None, None) is None  # Abstract body.
    conditional = ActionConditionalTransformerPredictor(slot_dim=8, action_dim=2, num_head=2, mlp_size=16)
    out = conditional(slots, torch.zeros(2, 2))
    assert out.shape == (2, 3, 8)
    assert not torch.allclose(out, conditional(slots, torch.ones(2, 2)))


def test_savi_properties_and_initialization():
    savi = make_tiny_savi()
    assert (savi.num_slots, savi.slot_dim) == (2, 8)
    assert torch.all(savi.corrector.gru.bias_ih == 0) and torch.all(savi.corrector.gru.bias_hh == 0)
    weight = savi.corrector.gru.weight_hh  # (3 * 8, 8), orthogonal columns.
    assert torch.allclose(weight.T @ weight, torch.eye(8), atol=1e-5)


def test_savi_initialization_of_slot_distribution_parameters():
    class CorrectorWithSlotDistribution(Corrector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dim_slots = self.slot_dim
            self.slots_mu = torch.nn.Parameter(torch.full((1, 1, self.slot_dim), 9.0))
            self.slots_sigma = torch.nn.Parameter(torch.full((1, 1, self.slot_dim), 9.0))

    savi = make_tiny_savi()
    savi.corrector = CorrectorWithSlotDistribution(2, 8, 8, 1, 2, 8)
    savi._initialize_parameters()
    limit = (6.0 / 9) ** 0.5
    assert savi.corrector.slots_mu.abs().max() <= limit
    assert savi.corrector.slots_sigma.abs().max() <= limit


def test_savi_encode_from_initializer_and_from_prior_slots():
    savi = make_tiny_savi(action_conditional=True).eval()
    images, actions = torch.rand(2, 3, 3, 16, 16), torch.randn(2, 2, 2)
    slots = savi.encode(images, actions)
    assert slots.shape == (2, 3, 2, 8)
    with mock.patch.object(savi.initializer, "forward", wraps=savi.initializer.forward) as initializer:
        continued = savi.encode(images[:, 2:], actions[:, :1], prior_slots=slots[:, 1])
        initializer.assert_not_called()
    assert continued.shape == (2, 1, 2, 8)
    # Without a prior the first frame uses the initial number of corrector iterations.
    with mock.patch.object(savi.corrector, "forward", wraps=savi.corrector.forward) as corrector:
        savi.encode(images[:, :2], actions[:, :1])
        assert [call.kwargs["is_first"] for call in corrector.call_args_list] == [True, False]
        savi.encode(images[:, :1], actions[:, :1], prior_slots=slots[:, 0])
        assert corrector.call_args_list[-1].kwargs["is_first"] is False


def test_savi_forward_and_decode():
    savi = make_tiny_savi()
    images = torch.rand(2, 3, 3, 16, 16)
    outputs = savi(images, torch.zeros(2, 2, 2))
    assert outputs["reconstructions"].shape == images.shape
    assert outputs["rgbs"].shape == (2, 3, 2, 3, 16, 16)
    assert outputs["masks"].shape == (2, 3, 2, 1, 16, 16)
    assert torch.allclose(
        outputs["reconstructions"], (outputs["rgbs"] * outputs["masks"]).sum(2).clamp(0, 1), atol=1e-6
    )


@pytest.mark.parametrize("with_images, with_ticks", [(True, True), (False, False)])
def test_savi_visualize_reconstruction(with_images, with_ticks):
    import numpy as np

    savi = make_tiny_savi()
    outputs = savi(torch.rand(1, 3, 3, 16, 16), torch.zeros(1, 2, 2))
    single = {k: v[0] for k, v in outputs.items()}
    single.pop("slots")
    if with_images:
        single["images"] = torch.rand(3, 3, 16, 16)
    if with_ticks:
        single["xticks"] = np.array(["a", "b", "c"])
    image = savi.visualize_reconstruction(single)
    rows = 1 + 1 + 2 + int(with_images)  # reconstructions, overlay, one per slot (+ images)
    expected_height = rows * 20 + (rows - 1) * 2
    if with_ticks:
        expected_height += 2 + (16 // 3 + 4)
    assert image.shape == (3, expected_height, 3 * 18 + 2)


# BUGS_AND_PITFALLS.md #17 (fixed).
def test_savi_initializes_slot_distribution_parameters():
    class CorrectorWithSlotDistribution(Corrector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.slots_mu = torch.nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            self.slots_sigma = torch.nn.Parameter(torch.zeros(1, 1, self.slot_dim))

    savi = make_tiny_savi()
    savi.corrector = CorrectorWithSlotDistribution(2, 8, 8, 1, 2, 8)
    savi._initialize_parameters()  # Reads the non-existent corrector.dim_slots.


# ----------------------------------------------------------------------------- logging callbacks
def _pl_module(log_dir, is_last_batch=True, epoch=0):
    module = mock.MagicMock()
    module.trainer.is_last_batch = is_last_batch
    module.current_epoch = epoch
    module.logger.log_dir = str(log_dir)
    return module


def test_logging_callback_should_log():
    callback = LoggingCallback(every_n_epochs=2)
    assert callback.should_log(_pl_module("x", True, 4))
    assert not callback.should_log(_pl_module("x", True, 3))
    assert not callback.should_log(_pl_module("x", False, 4))


def test_logging_callback_creates_save_dir(tmp_path):
    callback = LoggingCallback(save_dir="images")
    callback.on_fit_start(None, _pl_module(tmp_path))
    assert callback.save_dir == os.path.join(tmp_path, "images") and os.path.isdir(callback.save_dir)
    again = LoggingCallback(save_dir="images")
    again.on_fit_start(None, _pl_module(tmp_path))  # Already exists.
    none = LoggingCallback()
    none.on_fit_start(None, _pl_module(tmp_path))
    assert none.save_dir is None


def test_log_reconstruction_logs_first_sample_only_when_due():
    callback = LogReconstruction(every_n_epochs=1)
    module = _pl_module("x")
    outputs = {"images": torch.rand(2, 3, 3, 4, 4), "loss": torch.tensor(1.0)}
    callback.on_train_batch_end(None, module, outputs, None, 0)
    passed = module.autoencoder.visualize_reconstruction.call_args.args[0]
    assert set(passed) == {"images"} and passed["images"].shape == (3, 3, 4, 4)
    module.log_image.assert_called_once()
    module = _pl_module("x", is_last_batch=False)
    callback.on_train_batch_end(None, module, outputs, None, 0)
    module.log_image.assert_not_called()


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #31: Slot Attention scales by the wrong dimension")
def test_slot_attention_scales_by_the_query_key_dimension():
    corrector = Corrector(num_slots=7, slot_dim=128, feature_dim=64, num_iterations=1, num_initial_iterations=3)
    assert corrector.scale == pytest.approx(128**-0.5)  # q and k have slot_dim channels (paper Eq. 1).
