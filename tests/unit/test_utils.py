"""utils: module helpers, Hydra instantiation helpers, logging and visualizations."""

import json
import os
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from utils import visualizations as vis
from utils.instantiate import fill_in_missing, instantiate_dataloaders, instantiate_many, instantiate_trainer
from utils.logging import ExtendedTensorBoardLogger, LoggingStepMixin
from utils.module import FreezeParameters, get_parameters
from tests.conftest import write_image_dataset


# ----------------------------------------------------------------------------------- module
def test_get_parameters_and_freeze_restores_state():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    b.bias.requires_grad = False
    assert len(get_parameters([a, b])) == 4
    with FreezeParameters([a, b]):
        assert not any(p.requires_grad for p in get_parameters([a, b]))
    assert [p.requires_grad for p in get_parameters([a, b])] == [True, True, True, False]
    with pytest.raises(RuntimeError):
        with FreezeParameters([a]):
            raise RuntimeError("propagates")
    assert a.weight.requires_grad


# ------------------------------------------------------------------------------- instantiate
def test_instantiate_many_skips_non_targets():
    cfg = OmegaConf.create({"a": {"_target_": "builtins.dict", "x": 1}, "flag": False, "b": {"y": 2}})
    assert instantiate_many(cfg) == [{"x": 1}]


def test_instantiate_trainer(tmp_path):
    cfg = OmegaConf.create({
        "trainer": {"_target_": "lightning.Trainer", "max_epochs": 1, "accelerator": "cpu"},
        "logger": {"tb": {"_target_": "utils.logging.ExtendedTensorBoardLogger", "save_dir": str(tmp_path)},
                   "log_to_wandb": False},
        "callbacks": {"bar": {"_target_": "utils.training.OnlineProgressBar"}},
    })
    trainer = instantiate_trainer(cfg)
    assert isinstance(trainer.logger, ExtendedTensorBoardLogger)
    assert any(type(c).__name__ == "OnlineProgressBar" for c in trainer.callbacks)


def test_instantiate_dataloaders(tmp_path):
    write_image_dataset(tmp_path / "run", "npz")
    cfg = OmegaConf.create({"_target_": "datasets.image.load_image_dataset", "path": str(tmp_path), "data_dir": "run",
                            "sequence_length": 3, "batch_size": 2, "num_workers": 0})
    train, val, infos = instantiate_dataloaders(cfg)
    assert infos == {"image_size": [16, 16], "action_dim": 2}
    assert train.dataset.split == "train" and val.dataset.split == "val"
    images, actions = next(iter(train))
    assert images.shape == (2, 3, 3, 16, 16) and actions.shape == (2, 3, 2)


def test_fill_in_missing_by_key_suffix():
    cfg = OmegaConf.create({"model": {"encoder": {"image_size": "???"}, "decoder": {"image_size": "???"},
                                      "predictor": {"action_dim": "???"}, "other": "???"}})
    fill_in_missing(cfg, {"image_size": [8, 8], "action_dim": 3})
    assert cfg.model.encoder.image_size == [8, 8] and cfg.model.decoder.image_size == [8, 8]
    assert cfg.model.predictor.action_dim == 3
    assert OmegaConf.is_missing(cfg.model, "other")


# ----------------------------------------------------------------------------------- logging
def test_extended_tensorboard_logger_writes_metrics_images_and_videos(tmp_path):
    logger = ExtendedTensorBoardLogger(save_dir=str(tmp_path), name="logs")
    for sub in ["metrics", "images", "videos"]:
        assert os.path.isdir(os.path.join(logger.log_dir, sub))
    logger.log_metrics({"a": 1.0}, step=0)
    logger.log_metrics({"b": 2.0}, step=0)
    logger.log_metrics({"a": 3.0}, step=5)  # Flushes step 0.
    logger.log_metrics({"c": 4.0}, step=7)  # Flushes step 5.
    logger.current_step = None
    logger.log_metrics({"d": 5.0}, step=9)  # No flush when the current step is unknown.
    logger._flush_metrics()
    logger._flush_metrics()  # Nothing left to write.
    records = [json.loads(line) for line in open(os.path.join(logger.log_dir, "metrics", "metrics.jsonl"))]
    assert records == [{"step": 0, "a": 1.0, "b": 2.0}, {"step": 5, "a": 3.0}, {"step": 9, "c": 4.0, "d": 5.0}]

    logger.log_image("recon", torch.rand(3, 8, 8), step=3)
    assert os.path.isfile(os.path.join(logger.log_dir, "images", "recon-step=3.png"))
    logger.log_video("eval/episode_0", torch.rand(4, 3, 16, 16), step=3)
    logger.log_video("eval/raw", torch.randint(0, 255, (4, 3, 16, 16), dtype=torch.uint8), step=4)
    videos = sorted(os.listdir(os.path.join(logger.log_dir, "videos")))
    assert videos == ["eval_episode_0-step=3.mp4", "eval_raw-step=4.mp4"]


class _Recorder:
    """Stands in for LightningModule.log, the next class in the MRO."""

    def __init__(self):
        self.calls = []

    def log(self, name, value, *args, **kwargs):
        self.calls.append((name, value, kwargs))


class _Module(LoggingStepMixin, _Recorder):
    def __init__(self, logger, step=0, is_last_batch=True, max_logging_freq=1):
        LoggingStepMixin.__init__(self, max_logging_freq)
        _Recorder.__init__(self)
        self.logger = logger
        self.step = step
        self.trainer = mock.Mock(is_last_batch=is_last_batch)

    @property
    def logging_step(self):
        return self.step

    def _LightningModule__to_tensor(self, value, name):
        return torch.as_tensor(value)


def test_logging_step_mixin_routes_images_videos_and_scalars():
    logger = mock.MagicMock(spec=ExtendedTensorBoardLogger)
    module = _Module(logger, step=4)
    module.log("img", torch.zeros(3, 2, 2))
    logger.log_image.assert_called_once_with("img", mock.ANY, step=4)
    module.log("vid", torch.zeros(2, 3, 2, 2))
    logger.log_video.assert_called_once_with("vid", mock.ANY, step=4)
    module.log("matrix", torch.zeros(2, 2))  # Neither image nor video: dropped.
    assert module.calls == []

    module.log("loss", torch.tensor(0.5), prog_bar=True)
    logger.log_metrics.assert_called_once_with({"loss": 0.5}, step=4)
    assert module.calls == [("loss", 0.5, {"prog_bar": True, "logger": False})]
    module.log("loss", 0.25)  # Same step: throttled for the logger, still forwarded.
    assert logger.log_metrics.call_count == 1 and len(module.calls) == 2
    module.log("hidden", 1.0, logger=False)
    assert logger.log_metrics.call_count == 1
    module.trainer.is_last_batch = False
    module.step = 10
    module.log("loss", 2.0)
    assert logger.log_metrics.call_count == 1


def test_logging_step_mixin_other_loggers_and_default_step():
    module = _Module(logger=None)
    module.log("x", 1.0, prog_bar=True)
    assert module.calls == [("x", 1.0, {"prog_bar": True})]
    module.current_epoch = 3
    assert LoggingStepMixin.logging_step.fget(module) == 3


def test_log_gradients_logs_norm_of_available_grads():
    module = _Module(logger=None)
    module.net = torch.nn.Linear(2, 1)
    module.net.weight.grad = torch.full((1, 2), 3.0)
    module.net.bias.grad = None
    module.log_gradients(("net",))
    assert module.calls[0][0] == "gradients/net"
    assert module.calls[0][1] == pytest.approx(np.sqrt(18.0))


# ------------------------------------------------------------------------------ visualizations
def test_slot_color():
    assert torch.allclose(vis.slot_color(0, 3), torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(vis.slot_color(0, 1), vis.slot_color(0, 2))


def test_make_grid_layout_and_errors():
    images = torch.rand(5, 3, 4, 4)
    grid = vis.make_grid(images, num_columns=3, padding=1, pad_color=torch.tensor([1.0, 0.0, 0.0]))
    assert grid.shape == (3, 2 * 5 + 1, 3 * 5 + 1)
    assert torch.equal(grid[:, 1:5, 1:5], images[0])
    assert torch.equal(grid[:, 6:10, 6:10], images[4])
    assert grid[0, 0, 0] == 1 and grid[1, 0, 0] == 0
    assert vis.make_grid(images, num_columns=10).shape == (3, 8, 5 * 6 + 2)
    with pytest.raises(TypeError):
        vis.make_grid(np.zeros((1, 3, 2, 2)), 1)
    with pytest.raises(ValueError, match="pad_color"):
        vis.make_grid(images, 1, pad_color=torch.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        vis.make_grid(images[0], 1)
    with pytest.raises(ValueError, match="3 channels"):
        vis.make_grid(torch.rand(2, 1, 4, 4), 1)


def test_rows_columns_and_overlay():
    images = torch.rand(4, 3, 6, 6)
    assert vis.make_row(images).shape == (3, 10, 4 * 8 + 2)
    assert vis.make_row(images, max_sequence_length=2).shape == (3, 10, 2 * 8 + 2)
    stacked = vis.stack_rows([vis.make_row(images), vis.make_row(images)], height_spacing=3)
    assert stacked.shape == (3, 10 + 3 + 10, 34)
    columns = vis.stack_columns(images, num_context=1, width_spacing=2)
    assert columns.shape == (3, 10, 10 + 2 + 3 * 8 + 2)
    masks = torch.zeros(4, 2, 1, 6, 6)
    masks[:, 0] = 1.0
    overlay = vis.create_segmentation_overlay(images, masks, background_brightness=0.0)
    assert torch.allclose(overlay, vis.slot_color(0, 2).view(1, 3, 1, 1).expand(4, 3, 6, 6))


def test_draw_ticks_and_rewards():
    labels = vis.draw_ticks(torch.ones(2, 3, 8, 16), ["a", "b"], color=(0, 0, 0))
    assert labels.shape == (2, 3, 8, 16) and labels.max() <= 1.0 and labels.min() < 1.0
    drawn = vis.draw_reward(torch.zeros(2, 3, 32, 32), torch.tensor([0.5, 1.0]))
    assert drawn.dtype == torch.uint8 and drawn.shape == (2, 3, 32, 32) and drawn.max() > 200
    image = vis.visualize_reward_prediction(torch.zeros(3, 3, 32, 32), torch.zeros(3, 3, 32, 32),
                                            torch.tensor([float("nan"), 1.0, 2.0]), torch.tensor([0.0, 1.0, 2.0]))
    assert image.shape == (3, 2 * 36 + 2, 3 * 34 + 2)


def _predictor():
    from modeling.sold.prediction import TwoHotPredictor
    return TwoHotPredictor(max_episode_steps=6, num_slots=2, slot_dim=4, token_dim=8, num_heads=2, num_layers=2,
                           hidden_dim=8)


def test_get_attention_weights_on_a_predictor():
    model = _predictor()
    weights = vis.get_attention_weights(model, torch.randn(1, 3, 2, 4))
    assert weights.shape == (3, 2)
    assert weights.max().item() == pytest.approx(1.0)
    assert all(not m._forward_hooks for m in model.modules())


def test_attention_weights_hook_without_outputs():
    hook = vis.AttentionWeightsHook()
    hook(None, None, (None, torch.ones(1)))
    hook.clear()
    assert hook.outputs == [] and hook.compute_attention_weights("cpu", 2, 3) == 0

    class NoAttention(torch.nn.Module):
        def forward(self, slots, start=0):
            return slots

    assert vis.get_attention_weights(NoAttention(), torch.randn(1, 2, 2, 4)) == 0


def test_attention_visualizations():
    seq, slots = 3, 2
    weights = torch.rand(seq, slots)
    rgbs, masks = torch.rand(seq, slots, 3, 8, 8), torch.rand(seq, slots, 1, 8, 8)
    brightness, colormap = vis.get_output_attention_images(weights, rgbs, masks)
    assert brightness.shape == colormap.shape == (seq, 3, 8, 8)
    assert vis.visualize_output_attention(weights, torch.rand(seq, 3, 8, 8), rgbs, masks).shape == (3, 3 * 12 + 2 * 2,
                                                                                                   seq * 10 + 2)
    images, reconstructions = torch.rand(seq, 3, 32, 32), torch.rand(seq, 3, 32, 32)
    out = vis.visualize_reward_predictor_attention(images, reconstructions, torch.tensor([0.0, 1.0, 2.0]),
                                                   torch.tensor([0.0, 1.0, 2.5]), weights, torch.rand(seq, slots, 3, 32,
                                                                                                      32),
                                                   torch.rand(seq, slots, 1, 32, 32))
    assert out.shape == (3, 4 * 36 + 3 * 2, seq * 34 + 2)
