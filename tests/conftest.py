"""Shared fixtures. Tests run on CPU with tiny models; nothing here needs MuJoCo unless marked 'mujoco'."""

import os

os.environ.setdefault("HYDRA_FULL_ERROR", "1")

import functools
import matplotlib
from pathlib import Path
import random

matplotlib.use("Agg")

import numpy as np
import pytest
from tests import fakes  # noqa: F401  (registers FakePixel-v0)
import torch

REPO_DIR = Path(__file__).resolve().parents[1]
SOLD_DIR = REPO_DIR / "sold"
CONFIGS_DIR = REPO_DIR / "configs"
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"

torch.set_num_threads(4)


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def is_lfs_pointer(path: Path) -> bool:
    with open(path, "rb") as file:
        return file.read(40).startswith(b"version https://git-lfs")


def require_checkpoint(relative_path: str) -> Path:
    path = CHECKPOINTS_DIR / relative_path
    if not path.exists() or is_lfs_pointer(path):
        pytest.skip(f"{path} is missing or a Git LFS pointer; run 'git lfs pull'.")
    return path


# ------------------------------------------------------------------------------- tiny models
def make_tiny_savi(image_size=(16, 16), num_slots=2, slot_dim=8, action_dim=2, action_conditional=False):
    """A SAVi with the same structure as configs/autoencoder/savi.yaml, at toy size."""
    from modeling.autoencoder.savi import Corrector
    from modeling.autoencoder.savi.autoencoder import SAVi
    from modeling.autoencoder.savi.decoder import SaviCnnDecoder
    from modeling.autoencoder.savi.encoder import SaviCnnEncoder
    from modeling.autoencoder.savi.initializer import Learned
    from modeling.autoencoder.savi.predictor import ActionConditionalTransformerPredictor, TransformerPredictor

    predictor_class = ActionConditionalTransformerPredictor if action_conditional else TransformerPredictor
    return SAVi(
        corrector=Corrector(
            num_slots, slot_dim, feature_dim=8, num_iterations=1, num_initial_iterations=2, hidden_dim=8
        ),
        predictor=predictor_class(slot_dim, action_dim, 2, 8),
        encoder=SaviCnnEncoder(tuple(image_size), [4, 4], [3, 3], [1, 1], feature_dim=8),
        decoder=SaviCnnDecoder(tuple(image_size), [4, 4], [3, 3], [1, 1], in_channels=slot_dim),
        initializer=Learned(num_slots, slot_dim),
    )


def tiny_sold_kwargs(env, autoencoder=None, **overrides):
    from modeling.sold.dynamics import make_ocvp_seq_dynamics_model
    from modeling.sold.prediction import GaussianPredictor, TwoHotPredictor

    predictor = dict(token_dim=8, num_heads=2, num_layers=1, hidden_dim=16, num_mlp_layers=1)
    kwargs = dict(
        autoencoder=autoencoder if autoencoder is not None else make_tiny_savi(action_dim=env.action_space.shape[0]),
        dynamics_predictor=functools.partial(
            make_ocvp_seq_dynamics_model,
            token_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            residual=True,
            teacher_forcing=False,
        ),
        actor=functools.partial(GaussianPredictor, **predictor),
        critic=functools.partial(TwoHotPredictor, **predictor),
        reward_predictor=functools.partial(TwoHotPredictor, **predictor),
        optimization={
            "dynamics": {"learning_rate": 1e-3, "grad_clip": 3.0},
            "reward": {"learning_rate": 1e-3, "grad_clip": 10.0},
            "actor": {"learning_rate": 1e-3, "grad_clip": 10.0},
            "critic": {"learning_rate": 1e-3, "grad_clip": 10.0},
            "autoencoder": {"learning_rate": 1e-3, "grad_clip": 0.05},
        },
        finetune_autoencoder=False,
        num_context=2,
        imagination_horizon=3,
        start_imagination_from_every=False,
        actor_entropy_loss_weight=3e-4,
        actor_gradients="dynamics",
        return_lambda=0.95,
        discount_factor=0.96,
        critic_ema_decay=0.98,
        env=env,
        max_steps=40,
        num_seed=10,
        update_freq=5,
        num_updates=1,
        eval_freq=20,
        num_eval_episodes=2,
        batch_size=2,
        buffer_capacity=300,
        save_replay_buffer=False,
    )
    kwargs.update(overrides)
    return kwargs


def make_tiny_sold(env=None, **overrides):
    from train_sold import SOLDModule

    env = env if env is not None else fakes.make_fake_env()
    return SOLDModule(**tiny_sold_kwargs(env, **overrides))


@pytest.fixture
def fake_env():
    return fakes.make_fake_env()


@pytest.fixture
def tiny_savi():
    return make_tiny_savi()


@pytest.fixture
def tiny_sold(fake_env):
    return make_tiny_sold(fake_env)


def make_online_trainer(tmp_path, **kwargs):
    """A Lightning trainer configured like configs/trainer/online.yaml + callbacks/sold.yaml, on CPU."""
    from lightning import Trainer
    from utils.logging import ExtendedTensorBoardLogger
    from utils.training import OnlineProgressBar

    params = dict(
        max_steps=-1,
        max_epochs=-1,
        enable_checkpointing=False,
        devices=1,
        accelerator="cpu",
        logger=ExtendedTensorBoardLogger(save_dir=str(tmp_path), name="logs"),
        callbacks=[OnlineProgressBar()],
        enable_model_summary=False,
    )
    params.update(kwargs)
    return Trainer(**params)


# ----------------------------------------------------------------------------------- datasets
def make_episode(length, start=0, action_dim=2):
    """An episode in SOLD's format: obs[t], the action[t] that led to it and reward[t] (both NaN at t=0)."""
    obs = [torch.full((3, 2, 2), start + t, dtype=torch.uint8) for t in range(length)]
    action = [torch.full((action_dim,), float("nan"))] + [
        torch.full((action_dim,), float(start + t)) for t in range(1, length)
    ]
    reward = [torch.tensor(float("nan"))] + [torch.tensor(float(start + t)) for t in range(1, length)]
    return {"obs": obs, "action": action, "reward": reward}


def add_episode(buffer, episode):
    length = len(episode["obs"])
    for t in range(length):
        buffer.add_step({key: values[t] for key, values in episode.items()}, done=t == length - 1)


def write_image_dataset(
    root: Path,
    save_format: str = "npz",
    num_episodes=(3, 2),
    episode_length: int = 7,
    image_size=(16, 16),
    action_dim: int = 2,
) -> Path:
    """Write a dataset in the layout produced by generate_dataset.py: root/<split>/<episode>/..."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for split, count in zip(["train", "val"], num_episodes):
        for episode in range(count):
            episode_dir = root / split / str(episode)
            episode_dir.mkdir(parents=True)
            images = rng.integers(0, 256, (episode_length, *image_size, 3), dtype=np.uint8)
            data = {
                "actions": rng.uniform(-1, 1, (episode_length - 1, action_dim)).astype(np.float32),
                "rewards": rng.normal(size=episode_length - 1),
            }
            if save_format == "png":
                for index, image in enumerate(images):
                    Image.fromarray(image).save(episode_dir / f"{index}.png")
            else:
                data["images"] = images
            np.savez_compressed(episode_dir / "episode.npz", **data)
    return root
