"""Functions in the entry-point scripts: train_autoencoder, generate_dataset and evaluate_sold."""

import ast
import functools
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import evaluate_sold
import generate_dataset
from tests.conftest import CONFIGS_DIR, REPO_DIR, SOLD_DIR, make_tiny_savi, make_tiny_sold
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
    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, _loader(), _loader())
    assert trainer.callback_metrics["valid_loss"] == trainer.callback_metrics["valid/reconstruction_loss"]
    assert "train/reconstruction_loss" in trainer.callback_metrics
    path = tmp_path / "savi.ckpt"
    trainer.save_checkpoint(path)
    autoencoder = load_autoencoder(str(path))
    assert type(autoencoder).__name__ == "SAVi"
    for a, b in zip(autoencoder.state_dict().values(), module.autoencoder.state_dict().values()):
        assert torch.equal(a, b)


# BUGS_AND_PITFALLS.md #2 (fixed).
def test_savi_dataset_path_is_not_hardcoded():
    dataset = OmegaConf.load(CONFIGS_DIR / "train_autoencoder.yaml").dataset
    if OmegaConf.is_missing(dataset, "path") or OmegaConf.is_missing(dataset, "data_dir"):
        return  # Mandatory command-line values: nothing machine-specific in the config.
    configured = SOLD_DIR / dataset.path / dataset.data_dir
    assert configured.exists(), f"dataset points at {configured}, which only exists on the author's machine"


# BUGS_AND_PITFALLS.md #27 (fixed).
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to create a GPU checkpoint")
def test_savi_checkpoints_saved_on_gpu_load_on_cpu(tmp_path):
    module = AutoencoderModule(make_tiny_savi().cuda(), functools.partial(torch.optim.Adam))
    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, max_steps=0)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(tmp_path / "savi.ckpt")
    autoencoder = load_autoencoder(str(tmp_path / "savi.ckpt"))
    assert next(autoencoder.parameters()).device.type == "cpu", "a CPU-only machine could not load this checkpoint"


# ------------------------------------------------------------------------------- generate_dataset
def _dataset_cfg(save_format):
    return OmegaConf.create(
        {
            "save_format": save_format,
            "env": {
                "_target_": "tests.fakes.make_fake_env",
                "image_size": [16, 16],
                "max_episode_steps": 5,
                "action_repeat": 1,
            },
        }
    )


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


# BUGS_AND_PITFALLS.md #13 (fixed).
def test_savi_pretraining_uses_the_action_that_led_to_the_next_frame(tmp_path):
    from datasets.image import NPZDataset

    cfg = OmegaConf.create(
        {
            "save_format": "npz",
            "env": {"_target_": "tests.fakes.make_fake_env", "max_episode_steps": 6, "action_repeat": 1},
        }
    )
    (tmp_path / "run" / "val").mkdir(parents=True)
    generate_dataset.save_episode(str(tmp_path / "run" / "val" / "0"), cfg)
    taken = np.load(tmp_path / "run" / "val" / "0" / "episode.npz")["actions"]  # taken[t] moves frame t -> t+1.
    images, actions = NPZDataset(str(tmp_path), "run", "val", 4)[0]

    savi = make_tiny_savi(action_conditional=True)
    used = []
    original = savi.predictor.forward
    savi.predictor.forward = lambda slots, actions: (used.append(actions[0].clone()), original(slots, actions))[1]
    AutoencoderModule(savi, functools.partial(torch.optim.Adam)).compute_reconstruction_loss(
        images[None], actions[None]
    )
    for t in range(3):  # The predictor turns the slots of frame t into a prediction for frame t + 1.
        assert torch.allclose(used[t], torch.from_numpy(taken[t])), f"step {t} uses the action of step {t + 1}"


# --------------------------------------------------------------------------------- evaluate_sold
def test_get_checkpoint_files(tmp_path, monkeypatch):
    (tmp_path / "a.ckpt").write_text("x")
    (tmp_path / "b.ckpt").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    assert evaluate_sold.get_checkpoint_files(str(tmp_path / "a.ckpt")) == [str(tmp_path / "a.ckpt")]
    assert evaluate_sold.get_checkpoint_files(str(tmp_path / "notes.txt")) == []
    assert sorted(evaluate_sold.get_checkpoint_files(str(tmp_path))) == [
        str(tmp_path / "a.ckpt"),
        str(tmp_path / "b.ckpt"),
    ]
    with pytest.raises(ValueError, match="neither a valid file nor directory"):
        evaluate_sold.get_checkpoint_files(str(tmp_path / "missing"))
    monkeypatch.setattr(evaluate_sold.hydra.utils, "get_original_cwd", lambda: str(tmp_path))
    assert evaluate_sold.get_checkpoint_files("a.ckpt") == [str(tmp_path / "a.ckpt")]


def test_render_high_res(monkeypatch):
    env = make_tiny_sold().env
    render_sizes = []
    original_render = env.unwrapped.render
    monkeypatch.setattr(
        env.unwrapped,
        "render",
        lambda mode="rgb_array", size=None: render_sizes.append(size) or original_render(mode, (64, 48)),
    )
    frame = evaluate_sold.render_high_res(env)
    assert render_sizes == [(1024, 1024)]
    assert frame.shape == (3, 64, 48) and frame.dtype == torch.float32 and frame.max() <= 1.0


# BUGS_AND_PITFALLS.md #7 (fixed).
def test_evaluation_updates_the_last_action():
    sold = make_tiny_sold()
    seen = []
    original = sold.autoencoder.encode
    sold.autoencoder.encode = lambda images, actions, prior_slots=None: (
        seen.append(actions.clone()),
        original(images, actions, prior_slots),
    )[1]
    sold.runner.play_episode(mode="eval")  # As evaluate_sold.py plays its episodes.
    assert not torch.isnan(seen[-1]).any(), "the SAVi encoder still receives the initial NaN action"


# ------------------------------------------------------------------------------------ presets
_PRESET_COMMON = [
    "model.autoencoder.checkpoint_path=../checkpoints/autoencoder/savi/push_red/latest.ckpt",
    "model.actor.num_layers=4",
    "model.critic.num_layers=4",
    "model.reward_predictor.num_layers=4",
    "model.max_steps=20000000",
    "model.eval_freq=10000",
    "model.num_eval_episodes=30",
    "model.save_replay_buffer=False",
]


@pytest.mark.parametrize(
    "preset, overrides",
    [
        ("push_red", ["experiment=push_red_4layers_savi8", "environment@model.env=push_red"]),
        ("pick_red_frozen", ["experiment=pick_red_4layers_savi8_frozen", "environment@model.env=pick_red"]),
        (
            "pick_red_finetune",
            [
                "experiment=pick_red_4layers_savi8_finetune",
                "environment@model.env=pick_red",
                "model.finetune_autoencoder=True",
                "model.autoencoder_learning_rate=0.00003",
            ],
        ),
    ],
)
def test_presets_equal_the_overrides_of_the_training_scripts(preset, overrides):
    """configs/preset/ replaces the remote training scripts' command-line overrides."""
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        assert compose("train_sold", overrides=[f"preset={preset}"]) == compose(
            "train_sold", overrides=overrides + _PRESET_COMMON
        )


# ---------------------------------------------------------------------------------- all scripts
@pytest.mark.parametrize("environment", ["reach_red", "push_red", "pick_red"])
def test_training_and_evaluation_share_the_environment_configs(environment):
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        train = compose("train_sold", overrides=[f"environment@model.env={environment}"]).model.env
        evaluate = compose("evaluate_sold", overrides=[f"environment@env={environment}"]).env
    assert train == evaluate
    assert train.name.lower().startswith(environment.replace("_", ""))


# BUGS_AND_PITFALLS.md #16 (fixed).
def test_all_scripts_use_the_same_hydra_version_base():
    version_bases = {}
    for script in ["generate_dataset.py", "train_autoencoder.py", "train_sold.py", "evaluate_sold.py"]:
        tree = ast.parse((SOLD_DIR / script).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "main":
                keywords = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                version_bases[script] = keywords.get("version_base", "<not set>")
    assert len(set(version_bases.values())) == 1, version_bases


def test_importing_the_code_leaves_the_process_state_alone():
    """Importing modules must not change environment variables or warning filters; entry points do."""
    probe = """
import os, warnings
import gym, hydra, lightning, matplotlib, torch, torchvision  # Third-party imports may have their own effects.
environ, filters = dict(os.environ), list(warnings.filters)
import envs, evaluate_sold, generate_dataset, train_autoencoder, train_sold  # noqa: F401
assert dict(os.environ) == environ, set(os.environ.items()) ^ set(environ.items())
assert list(warnings.filters) == filters, [f for f in warnings.filters if f not in filters]
"""
    environ = {k: v for k, v in os.environ.items() if k not in ("HYDRA_FULL_ERROR", "MUJOCO_GL")}
    environ["PYTHONPATH"] = str(SOLD_DIR)  # The repository root is the working directory.
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_DIR, env=environ, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
