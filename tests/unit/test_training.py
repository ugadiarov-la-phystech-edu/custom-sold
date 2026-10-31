"""utils/training.py: seeding, summary, the online progress bar and the OnlineModule collection loop."""

import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from tests.conftest import add_episode, make_episode, make_online_trainer, make_tiny_sold
from tests import fakes
from utils import training
from utils.episode_runner import EpisodeRunner
from utils.training import OnlineProgressBar, print_summary, set_seed


def test_set_seed_is_reproducible(monkeypatch):
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    set_seed(3)
    first = draw()
    set_seed(3)
    assert draw() == first
    assert os.environ["PYTHONHASHSEED"] == "3"

    cuda_seed = mock.Mock()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed", cuda_seed)
    set_seed(5)
    cuda_seed.assert_called_once_with(5)


def test_print_summary(monkeypatch, capsys):
    hydra_config = SimpleNamespace(runtime=SimpleNamespace(output_dir="/out/dir"))
    monkeypatch.setattr(training.hydra.core.hydra_config.HydraConfig, "get", staticmethod(lambda: hydra_config))
    cfg = SimpleNamespace(model=SimpleNamespace(env=SimpleNamespace(name="PushRed"), max_steps=1234567), experiment="exp")
    print_summary(cfg)
    out = capsys.readouterr().out
    for text in ["Output Dir:", "/out/dir", "Task:", "PushRed", "Max steps:", "1,234,567", "Experiment:", "exp"]:
        assert text in out


def test_progress_bar_helpers():
    bar = OnlineProgressBar()
    module = SimpleNamespace(num_steps=12, num_episodes=-1, max_steps=99)
    bar.pl_module = module
    assert bar._get_train_description(0) == "Steps 12, Episodes 0"
    assert bar.on_train_epoch_start(None, module) is None
    assert bar.total_train_batches == 99


def test_online_module_initial_state(tiny_sold):
    assert tiny_sold.num_steps == 0 and tiny_sold.num_episodes == -1
    assert tiny_sold.last_action.shape == (2,) and torch.isnan(tiny_sold.last_action).all()
    assert tiny_sold.sequence_length == 3 + 2
    assert tiny_sold.logging_step == 0
    tiny_sold.num_steps = 7
    assert tiny_sold.logging_step == 7


def test_complete_first_timestep_adds_nan_action_and_reward(tiny_sold):
    step = tiny_sold._complete_first_timestep({"obs": torch.zeros(1)})
    assert torch.isnan(step["action"]).all() and step["action"].shape == (2,) and torch.isnan(step["reward"])
    given = {"obs": 0, "action": torch.ones(2), "reward": torch.tensor(1.0)}
    assert tiny_sold._complete_first_timestep(dict(given))["reward"] == 1.0


def test_online_module_runner_updates_the_last_action(tiny_sold):
    tiny_sold.runner.play_episode(mode="eval")
    assert not torch.isnan(tiny_sold.last_action).any()  # select_action reads it to encode the next observation.


# ----------------------------------------------------------------------------- episode runner
def _runner(**env_kwargs):
    env = fakes.make_fake_env(**env_kwargs)
    calls = []

    def policy(obs, is_first, mode):
        calls.append((is_first, mode))
        return torch.full((2,), 0.5)

    return EpisodeRunner(env, policy), calls


def test_episode_runner_steps_and_marks_the_first_step():
    runner, calls = _runner()
    assert runner.done and runner.obs is None
    first = runner.reset()
    assert torch.equal(first, runner.obs) and not runner.done
    runner.step("random")
    obs, reward, done, info = runner.step("train")
    assert calls == [(True, "random"), (False, "train")]
    assert torch.equal(obs, runner.obs) and not done
    runner.reset()
    assert runner.is_first


def test_episode_runner_plays_whole_episodes_with_rendering():
    runner, calls = _runner()
    episode = runner.play_episode("eval", render=lambda env: torch.zeros(3, 4, 4))
    assert len(episode["obs"]) == len(episode["rendered"]) == 9 and len(episode["reward"]) == 8
    assert "success" in episode and runner.done
    assert calls[0] == (True, "eval") and all(not is_first for is_first, _ in calls[1:])
    assert "rendered" not in runner.play_episode("train")


def test_episode_runner_requires_a_finished_episode():
    runner, _ = _runner()
    runner.reset()
    with pytest.raises(RuntimeError, match="must have terminated"):
        runner.play_episode()


def test_episode_runner_without_success_info(monkeypatch):
    runner, _ = _runner()
    original_step = runner.env.unwrapped.step
    monkeypatch.setattr(runner.env.unwrapped, "step", lambda action: original_step(action)[:3] + ({},))
    assert "success" not in runner.play_episode(mode="train")


def test_on_train_epoch_end_stops_at_max_steps(tiny_sold):
    tiny_sold.trainer = SimpleNamespace(should_stop=False)
    tiny_sold.num_steps = tiny_sold.max_steps - 1
    tiny_sold.on_train_epoch_end()
    assert tiny_sold.trainer.should_stop is False
    tiny_sold.num_steps = tiny_sold.max_steps
    tiny_sold.on_train_epoch_end()
    assert tiny_sold.trainer.should_stop is True


def test_on_load_checkpoint_without_replay_buffer(tiny_sold, tmp_path):
    tiny_sold.on_load_checkpoint({"num_steps": 300, "num_episodes": 12})
    assert (tiny_sold.num_steps, tiny_sold.num_episodes) == (300, 11)
    tiny_sold.on_load_checkpoint({"num_steps": 5, "num_episodes": 1, "replay_buffer_path": tmp_path / "gone"})
    assert tiny_sold.num_steps == 5 and not hasattr(tiny_sold, "replay_buffer")


def test_on_load_checkpoint_rejects_buffer_without_fields(tiny_sold, tmp_path):
    from datasets.info import EpisodeDatasetInfo
    tiny_sold._trainer = mock.Mock()
    with mock.patch.object(type(tiny_sold), "logger", new_callable=mock.PropertyMock,
                           return_value=SimpleNamespace(log_dir=str(tmp_path))):
        with pytest.raises(ValueError, match="without fields"):
            tiny_sold.on_load_checkpoint({"num_steps": 5, "num_episodes": 1, "replay_buffer_path": tmp_path,
                                          "replay_buffer_info": EpisodeDatasetInfo()})


def _fit(tmp_path, save_replay_buffer=True, **overrides):
    sold = make_tiny_sold(save_replay_buffer=save_replay_buffer, **overrides)
    trainer = make_online_trainer(tmp_path)
    trainer.fit(sold)
    return sold, trainer


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fit")
    torch.manual_seed(0)
    sold, trainer = _fit(tmp_path)
    return sold, trainer, Path(trainer.logger.log_dir)


def test_fit_collects_until_max_steps(fitted):
    sold, trainer, _ = fitted
    assert sold.num_steps == 40 and trainer.should_stop
    # 40 steps of 8-step episodes, plus the initial reset.
    assert sold.num_episodes == 4
    assert sold.replay_buffer.num_episodes == 5 and sold.replay_buffer.num_timesteps == 45
    assert trainer.global_step > 0


def test_fit_evaluates_and_checkpoints(fitted):
    _, _, log_dir = fitted
    checkpoints = sorted(p.name for p in (log_dir / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 2
    assert checkpoints[0].startswith("sold-steps=0-episode=0-eval_episode_return=")
    # Evaluation is due at step 20 but waits for the episode to end (step 24).
    assert checkpoints[1].startswith("sold-steps=24-episode=3-eval_episode_return=")
    videos = os.listdir(log_dir / "videos")
    assert {"eval_episode_0-step=0.mp4", "eval_episode_1-step=0.mp4", "train_episode_2-step=24.mp4"} <= set(videos)


def test_fit_keeps_only_the_latest_replay_buffer_copy(fitted):
    _, _, log_dir = fitted
    copies = sorted(p.name for p in (log_dir / "checkpoints").glob("replay_buffer-steps=*"))
    assert copies == ["replay_buffer-steps=24-episode=3"]
    assert sorted(os.listdir(log_dir / "checkpoints" / copies[0])) == ["action.dat", "obs.dat", "reward.dat"]
    checkpoint = torch.load(sorted((log_dir / "checkpoints").glob("sold-steps=24*.ckpt"))[0], weights_only=False)
    assert checkpoint["num_steps"] == 24 and checkpoint["num_episodes"] == 3
    assert Path(checkpoint["replay_buffer_path"]).name == copies[0]
    assert checkpoint["replay_buffer_capacity"] == 300 and checkpoint["replay_buffer_info"].num_episodes == 3


def test_fit_logs_metrics(fitted):
    _, _, log_dir = fitted
    records = [json.loads(line) for line in open(log_dir / "metrics" / "metrics.jsonl")]
    keys = set().union(*records)
    assert {"eval/episode_return", "eval/success_rate", "train/episode_return", "train/buffer_size",
            "train/dynamics_loss", "train/reward_loss", "train/actor_loss", "train/critic_loss",
            "gradients/actor"} <= keys
    assert [r["step"] for r in records] == sorted(r["step"] for r in records)
    images = os.listdir(log_dir / "images")
    for name in ["autoencoder_reconstruction", "dynamics_prediction", "reward_prediction",
                 "reward_predictor_attention", "latent_imagination", "actor_attention"]:
        assert any(image.startswith(name + "-step=") for image in images), name


def test_fit_progress_bar_is_created(fitted):
    _, trainer, _ = fitted
    bar = next(c for c in trainer.callbacks if isinstance(c, OnlineProgressBar))
    assert bar.progress is not None or not bar.is_enabled


def test_fit_without_saving_replay_buffer_and_with_finetuning(tmp_path):
    sold, trainer = _fit(tmp_path, save_replay_buffer=False, finetune_autoencoder=True, max_steps=20,
                         start_imagination_from_every=True, num_context=(1, 2))
    assert sold.num_steps == 20
    assert not list(Path(trainer.logger.log_dir, "checkpoints").glob("replay_buffer-*"))
    checkpoint = torch.load(next(Path(trainer.logger.log_dir, "checkpoints").glob("*.ckpt")), weights_only=False)
    assert "replay_buffer_path" not in checkpoint


def test_save_replay_buffer_when_buffer_folder_is_missing(tiny_sold, tmp_path):
    tiny_sold.save_replay_buffer = True
    tiny_sold.replay_buffer = SimpleNamespace(save_path=str(tmp_path / "missing"), capacity=1, info=None, head=0,
                                              tail=0, episode_boundaries=[])
    with mock.patch.object(type(tiny_sold), "checkpoints_dir", new_callable=mock.PropertyMock,
                           return_value=tmp_path):
        checkpoint = {}
        tiny_sold.on_save_checkpoint(checkpoint)
    assert checkpoint["replay_buffer_path"] == tmp_path / "replay_buffer-steps=0-episode=-1"
    assert not checkpoint["replay_buffer_path"].exists()


def test_checkpoints_dir_is_created_under_log_dir(tiny_sold, tmp_path):
    with mock.patch.object(type(tiny_sold), "logger", new_callable=mock.PropertyMock,
                           return_value=SimpleNamespace(log_dir=str(tmp_path))):
        assert tiny_sold.checkpoints_dir == tmp_path / "checkpoints" and (tmp_path / "checkpoints").is_dir()


def test_abstract_select_action_body():
    assert training.OnlineModule.select_action(None, None) is None


def test_save_replay_buffer_replaces_the_previous_copy(tiny_sold, tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "obs.dat").write_bytes(b"new")
    old = tmp_path / "checkpoints" / "replay_buffer-steps=5-episode=1"
    old.mkdir(parents=True)
    tiny_sold.save_replay_buffer, tiny_sold.num_steps, tiny_sold.num_episodes = True, 7, 2
    tiny_sold.replay_buffer = SimpleNamespace(save_path=str(live), capacity=1, info=None, head=0, tail=0,
                                              episode_boundaries=[])
    with mock.patch.object(type(tiny_sold), "checkpoints_dir", new_callable=mock.PropertyMock,
                           return_value=tmp_path / "checkpoints"):
        tiny_sold.on_save_checkpoint({})
    assert [p.name for p in (tmp_path / "checkpoints").iterdir()] == ["replay_buffer-steps=7-episode=2"]
    assert (tmp_path / "checkpoints" / "replay_buffer-steps=7-episode=2" / "obs.dat").read_bytes() == b"new"


def test_on_load_checkpoint_without_trainer_ignores_the_replay_buffer(tiny_sold, tmp_path):
    (tmp_path / "saved").mkdir()  # Exists, but there is no trainer (and no logger directory) to restore into (#6).
    tiny_sold.on_load_checkpoint({"num_steps": 5, "num_episodes": 1, "replay_buffer_path": tmp_path / "saved"})
    assert tiny_sold.num_steps == 5 and not hasattr(tiny_sold, "replay_buffer")


# BUGS_AND_PITFALLS.md #3a (fixed).
def test_progress_bar_creates_its_display_on_train_start(tiny_sold):
    from lightning import Trainer
    bar = OnlineProgressBar()
    trainer = Trainer(accelerator="cpu", logger=False, callbacks=[bar], enable_model_summary=False)
    bar.setup(trainer, tiny_sold, stage="fit")
    bar.on_train_start(trainer, tiny_sold)
    # Lightning >= 2.5.3 calls on_train_batch_end -> _add_task, which asserts that the display exists.
    assert bar.progress is not None


# BUGS_AND_PITFALLS.md #4 (fixed).
def test_resuming_restores_the_saved_replay_buffer(tiny_sold, tmp_path):
    from datasets.ring_buffer import RingBufferDataset
    saved = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "saved"))
    add_episode(saved, make_episode(5))
    checkpoint = {"num_steps": 5, "num_episodes": 1, "replay_buffer_path": tmp_path / "saved",
                  "replay_buffer_capacity": 20, "replay_buffer_info": saved.info, "replay_buffer_head": saved.head,
                  "replay_buffer_tail": saved.tail, "replay_buffer_episode_boundaries": saved.episode_boundaries}
    tiny_sold._trainer = mock.Mock()  # Resuming happens inside trainer.fit (see #6 for the case without one).
    with mock.patch.object(type(tiny_sold), "logger", new_callable=mock.PropertyMock,
                           return_value=SimpleNamespace(log_dir=str(tmp_path / "run"))):
        tiny_sold.on_load_checkpoint(checkpoint)
    assert tiny_sold.replay_buffer.num_episodes == 1 and tiny_sold.replay_buffer.tail == 5
    assert np.array_equal(tiny_sold.replay_buffer.ring_buffer["obs"], saved.ring_buffer["obs"])


# BUGS_AND_PITFALLS.md #5 (fixed).
def test_pruning_keeps_the_newest_replay_buffer(tiny_sold, tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "replay_buffer-steps=9000-episode=180").mkdir(parents=True)
    live = tmp_path / "live"
    live.mkdir()
    (live / "obs.dat").write_bytes(b"x")
    tiny_sold.save_replay_buffer, tiny_sold.num_steps, tiny_sold.num_episodes = True, 10500, 210
    tiny_sold.replay_buffer = SimpleNamespace(save_path=str(live), capacity=1, info=None, head=0, tail=0,
                                              episode_boundaries=[])
    checkpoint = {}
    with mock.patch.object(type(tiny_sold), "checkpoints_dir", new_callable=mock.PropertyMock,
                           return_value=checkpoints_dir):
        tiny_sold.on_save_checkpoint(checkpoint)
    assert [p.name for p in checkpoints_dir.iterdir()] == ["replay_buffer-steps=10500-episode=210"]
    assert checkpoint["replay_buffer_path"].exists()


# BUGS_AND_PITFALLS.md #6 (fixed).
def test_latest_checkpoint_loads_without_a_trainer(fitted):
    from train_sold import SOLDModule
    _, _, log_dir = fitted
    checkpoints_dir = log_dir / "checkpoints"
    assert list(checkpoints_dir.glob("replay_buffer-*")), "precondition: the replay buffer copy exists"
    latest = sorted(checkpoints_dir.glob("*.ckpt"))[-1]
    loaded = SOLDModule.load_from_checkpoint(latest, env=fakes.make_fake_env())  # As evaluate_sold.py does.
    assert loaded.num_steps == torch.load(latest, weights_only=False)["num_steps"]
