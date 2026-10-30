"""Functions in the entry-point scripts: train_autoencoder, generate_dataset and evaluate_sold."""

import functools
import os

import numpy as np
import pytest
import torch
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import evaluate_sold
import generate_dataset
from tests.conftest import make_tiny_savi, make_tiny_sold
from train_autoencoder import AutoencoderModule, load_autoencoder


# ------------------------------------------------------------------------------ train_autoencoder
def _autoencoder_module(scheduler=None):
    return AutoencoderModule(make_tiny_savi(), functools.partial(torch.optim.Adam, lr=1e-3), scheduler)


def test_configure_optimizers_without_scheduler():
    config = _autoencoder_module().configure_optimizers()
    assert set(config) == {"optimizer"} and config["optimizer"].param_groups[0]["lr"] == 1e-3


@pytest.mark.parametrize("extras", [None, {"interval": "epoch", "frequency": 2}])
def test_configure_optimizers_with_scheduler(extras):
    # As produced by Hydra from configs/train_autoencoder.yaml: a config node holding an instantiated partial.
    scheduler = OmegaConf.create({"extras": extras}, flags={"allow_objects": True})
    scheduler.scheduler = functools.partial(torch.optim.lr_scheduler.StepLR, step_size=1)
    config = _autoencoder_module(scheduler).configure_optimizers()
    assert isinstance(config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.StepLR)
    assert config["lr_scheduler"] == {"scheduler": config["lr_scheduler"]["scheduler"], **(extras or {})}


def test_compute_reconstruction_loss_skips_first_action():
    module = _autoencoder_module()
    images, actions = torch.rand(2, 3, 3, 16, 16), torch.rand(2, 3, 2)
    with torch.no_grad():
        outputs = module.compute_reconstruction_loss(images, actions)
    assert torch.equal(outputs["images"], images)
    assert torch.allclose(outputs["reconstruction_loss"], torch.mean((outputs["reconstructions"] - images) ** 2))
    assert {"slots", "rgbs", "masks", "reconstructions"} <= set(outputs)


def _loader():
    return DataLoader(TensorDataset(torch.rand(4, 3, 3, 16, 16), torch.rand(4, 3, 2)), batch_size=2)


def test_autoencoder_fit_and_load_checkpoint(tmp_path):
    module = _autoencoder_module()
    trainer = Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(module, _loader(), _loader())
    assert trainer.callback_metrics["valid_loss"] == trainer.callback_metrics["valid/reconstruction_loss"]
    assert "train/reconstruction_loss" in trainer.callback_metrics
    path = tmp_path / "savi.ckpt"
    trainer.save_checkpoint(path)
    autoencoder = load_autoencoder(str(path))
    assert type(autoencoder).__name__ == "SAVi"
    for a, b in zip(autoencoder.state_dict().values(), module.autoencoder.state_dict().values()):
        assert torch.equal(a, b)


# ------------------------------------------------------------------------------- generate_dataset
def _dataset_cfg(save_format):
    return OmegaConf.create({"save_format": save_format,
                             "env": {"_target_": "tests.fakes.make_fake_env", "image_size": [16, 16],
                                     "max_episode_steps": 5, "action_repeat": 1}})


def test_save_episode_png(tmp_path):
    generate_dataset.save_episode(str(tmp_path / "0"), _dataset_cfg("png"))
    files = sorted(os.listdir(tmp_path / "0"))
    assert files == ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "episode.npz"]
    data = np.load(tmp_path / "0" / "episode.npz")
    assert set(data.files) == {"actions", "rewards"}
    assert data["actions"].shape == (5, 2) and data["rewards"].shape == (5,)


def test_save_episode_npz(tmp_path):
    generate_dataset.save_episode(str(tmp_path / "0"), _dataset_cfg("npz"))
    assert os.listdir(tmp_path / "0") == ["episode.npz"]
    data = np.load(tmp_path / "0" / "episode.npz")
    assert data["images"].shape == (6, 16, 16, 3) and data["images"].dtype == np.uint8
    assert data["actions"].dtype == np.float32 and np.abs(data["actions"]).max() <= 1


def test_save_episode_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported save format"):
        generate_dataset.save_episode(str(tmp_path / "0"), _dataset_cfg("gif"))
    with pytest.raises(FileExistsError):  # Refuses to overwrite an episode folder.
        generate_dataset.save_episode(str(tmp_path / "0"), _dataset_cfg("npz"))


# --------------------------------------------------------------------------------- evaluate_sold
def test_get_checkpoint_files(tmp_path, monkeypatch):
    (tmp_path / "a.ckpt").write_text("x")
    (tmp_path / "b.ckpt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    assert evaluate_sold.get_checkpoint_files(str(tmp_path / "a.ckpt")) == [str(tmp_path / "a.ckpt")]
    assert evaluate_sold.get_checkpoint_files(str(tmp_path / "notes.txt")) == []
    assert sorted(evaluate_sold.get_checkpoint_files(str(tmp_path))) == [str(tmp_path / "a.ckpt"),
                                                                          str(tmp_path / "b.ckpt")]
    with pytest.raises(ValueError, match="neither a valid file nor directory"):
        evaluate_sold.get_checkpoint_files(str(tmp_path / "missing"))
    monkeypatch.setattr(evaluate_sold.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    assert evaluate_sold.get_checkpoint_files("a.ckpt") == [str(tmp_path / "a.ckpt")]


def test_evaluate_play_episode_records_high_resolution_frames(monkeypatch):
    sold = make_tiny_sold()
    render_sizes = []
    original_render = sold.env.unwrapped.render
    monkeypatch.setattr(sold.env.unwrapped, "render",
                        lambda mode="rgb_array", size=None: render_sizes.append(size) or original_render(mode, (64, 48)))
    episode = evaluate_sold.play_episode(sold)
    assert len(episode["obs"]) == len(episode["high_res"]) == 9 and len(episode["reward"]) == 8
    assert render_sizes.count((1024, 1024)) == 9  # Observations themselves are rendered without a size.
    assert episode["high_res"][0].shape == (3, 64, 48) and episode["high_res"][0].max() <= 1.0
    assert "success" in episode


def test_evaluate_play_episode_without_success(monkeypatch):
    sold = make_tiny_sold()
    original_step = sold.env.unwrapped.step
    monkeypatch.setattr(sold.env.unwrapped, "step", lambda action: original_step(action)[:3] + ({},))
    assert "success" not in evaluate_sold.play_episode(sold, mode="train")
