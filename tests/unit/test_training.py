"""utils/training.py: seeding, summary, the progress bar and the OnlineModule training loop."""

from collections import deque
import copy
import logging
import numpy as np
import os
from pathlib import Path
import pytest
import random
from sold.data.ring_buffer import RingBufferDataset
from sold.utils import training
from sold.utils.episode_runner import EpisodeRunner
from sold.utils.profiling import Timings
from sold.utils.training import ProgressBar, print_summary, resolve_device, set_seed
from tests import fakes
from tests.conftest import add_episode, fit_sold, make_episode, make_tiny_sold, read_metrics
from tests.unit.test_profiling import FakeEvent
import torch
from types import SimpleNamespace
from unittest import mock


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


def test_resolve_device(monkeypatch):
    assert resolve_device("cpu") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")


def test_print_summary(monkeypatch, capsys):
    hydra_config = SimpleNamespace(runtime=SimpleNamespace(output_dir="/out/dir"))
    monkeypatch.setattr(training.hydra.core.hydra_config.HydraConfig, "get", staticmethod(lambda: hydra_config))
    cfg = SimpleNamespace(
        model=SimpleNamespace(env=SimpleNamespace(name="PushRed"), max_steps=1234567), experiment="exp"
    )
    print_summary(cfg)
    out = capsys.readouterr().out
    for text in ["Output Dir:", "/out/dir", "Task:", "PushRed", "Max steps:", "1,234,567", "Experiment:", "exp"]:
        assert text in out


def test_progress_bar_describes_steps_episodes_and_eval_return():
    with ProgressBar(total=99, enabled=False) as bar:
        bar.update(12, -1)
        task = bar.progress.tasks[0]
        assert (task.description, task.completed, task.total) == ("Steps 12, Episodes 0", 12, 99)
        bar.eval_return = 1.5
        bar.update(21, 2)
        assert bar.progress.tasks[0].description == "Steps 21, Episodes 2, eval return 1.50"
    assert ProgressBar(total=1).progress.disable  # No terminal under pytest.
    assert not ProgressBar(total=1, enabled=True).progress.disable


def test_online_module_initial_state(tiny_sold):
    assert tiny_sold.num_steps == 0 and tiny_sold.num_episodes == -1
    assert tiny_sold.last_action.shape == (2,) and torch.isnan(tiny_sold.last_action).all()
    assert "last_action" not in tiny_sold.state_dict()  # Not persistent: the checkpoint keys stay as they were.
    assert tiny_sold.sequence_length == 3 + 2
    assert tiny_sold.device == torch.device("cpu") and tiny_sold.replay_buffer is None
    assert {"max_steps", "optimization", "profile", "autoencoder"} <= set(tiny_sold.hparams)
    assert "env" not in tiny_sold.hparams


def test_online_module_runner_updates_the_last_action(tiny_sold):
    tiny_sold.runner.play_episode(mode="eval")
    assert not torch.isnan(tiny_sold.last_action).any()  # select_action reads it to encode the next observation.


# ----------------------------------------------------------------------------- episode runner
def _runner(timings=None, **env_kwargs):
    env = fakes.make_fake_env(**env_kwargs)
    calls = []

    def policy(obs, is_first, mode):
        calls.append((is_first, mode))
        return torch.full((2,), 0.5)

    return EpisodeRunner(env, policy, timings), calls


def test_episode_runner_times_collection_and_evaluation_separately():
    timings = Timings()
    runner, _ = _runner(timings)
    runner.reset()
    runner.step("train")
    assert set(timings.summary()) == {"collect/env_reset", "collect/action_selection", "collect/env_step"}
    runner.done = True
    runner.play_episode()
    assert set(timings.summary()) == {"eval/env_reset", "eval/action_selection", "eval/env_step"}


def test_episode_runner_steps_and_marks_the_first_step():
    runner, calls = _runner()
    assert runner.done and runner.obs is None
    first = runner.reset()
    assert torch.equal(first, runner.obs) and not runner.done
    runner.step("random")
    obs, reward, done, info = runner.step("train")
    assert torch.equal(runner.action, torch.full((2,), 0.5))
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


# ----------------------------------------------------------------------------- round logging
def _with_fake_logger(tiny_sold, num_steps=30):
    tiny_sold.logger = mock.Mock()
    tiny_sold.num_steps = num_steps
    return tiny_sold.logger


def test_log_round_writes_losses_gradient_norms_and_timings(tiny_sold):
    logger = _with_fake_logger(tiny_sold)
    expected_norms = {}
    for name in tiny_sold.GRADIENT_NORMS:
        param = next(getattr(tiny_sold, name).parameters())
        param.grad = torch.full_like(param, 3.0)
        expected_norms[f"gradients/{name}"] = 3.0 * np.sqrt(param.numel())
    tiny_sold.log_round({"actor_loss": torch.tensor(0.5), "predicted_rewards": torch.zeros(1)})
    (metrics,), kwargs = logger.log_metrics.call_args
    assert kwargs == {"step": 30}
    assert metrics["train/actor_loss"] == 0.5 and "train/predicted_rewards" not in metrics
    assert {name: norm.item() for name, norm in metrics.items() if name.startswith("gradients/")} == pytest.approx(
        expected_norms
    )
    assert "timing/learn/logging_ms" in metrics and metrics["timing/env_steps_per_second"] > 0
    assert not any(name.startswith("memory/") for name in metrics)


def test_log_round_without_profiling_writes_no_timings(tiny_sold):
    logger = _with_fake_logger(tiny_sold)
    tiny_sold.timings.enabled = False
    tiny_sold.GRADIENT_NORMS = ()
    tiny_sold.log_round({"critic_loss": torch.tensor(1.0)})
    assert logger.log_metrics.call_args.args[0] == {"train/critic_loss": torch.tensor(1.0)}


def test_log_round_on_a_gpu_drains_it_and_logs_peak_memory(tiny_sold, monkeypatch):
    logger = _with_fake_logger(tiny_sold)
    synchronized, reset = [], []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2**30)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2**31)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset.append)
    monkeypatch.setattr(type(tiny_sold), "device", property(lambda self: torch.device("cuda")))
    tiny_sold.timings.event_factory = FakeEvent
    tiny_sold.GRADIENT_NORMS = ()
    tiny_sold.log_round({})
    metrics = logger.log_metrics.call_args.args[0]
    assert [device.type for device in synchronized] == ["cuda"] and len(reset) == 1
    assert metrics["memory/gpu_peak_allocated_gb"] == 1.0 and metrics["memory/gpu_peak_reserved_gb"] == 2.0
    assert {"timing/learn/gpu_drain_ms", "timing/learn/logging_ms"} <= set(metrics)


# ----------------------------------------------------------------------------- training runs
def _fit(tmp_path, save_replay_buffer=True, **overrides):
    sold = make_tiny_sold(save_replay_buffer=save_replay_buffer, **overrides)
    logger = fit_sold(sold, tmp_path)
    return sold, logger


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fit")
    torch.manual_seed(0)
    sold, logger = _fit(tmp_path)
    return sold, Path(logger.log_dir)


def test_fit_collects_until_max_steps(fitted):
    sold, _ = fitted
    assert sold.num_steps == 40
    # 40 steps of 8-step episodes, plus the initial reset.
    assert sold.num_episodes == 4
    assert sold.replay_buffer.num_episodes == 5 and sold.replay_buffer.num_timesteps == 45
    assert len(sold.optimizers) == 4


def test_fit_stops_after_the_seed_round_when_max_steps_is_reached(tmp_path):
    sold, logger = _fit(tmp_path, max_steps=10)  # num_seed is 10: no round of updates follows.
    assert sold.num_steps == 10
    assert not any("train/dynamics_loss" in record for record in read_metrics(logger.log_dir))


def test_fit_evaluates_and_checkpoints(fitted):
    _, log_dir = fitted
    checkpoints = sorted(p.name for p in (log_dir / "checkpoints").glob("*.ckpt"))
    assert len(checkpoints) == 2
    assert checkpoints[0].startswith("sold-steps=0-episode=0-eval_episode_return=")
    # Evaluation is due at step 20 but waits for the episode to end (step 24).
    assert checkpoints[1].startswith("sold-steps=24-episode=3-eval_episode_return=")
    videos = os.listdir(log_dir / "videos")
    assert {"eval_episode_0-step=0.mp4", "eval_episode_1-step=0.mp4", "train_episode_2-step=24.mp4"} <= set(videos)


def test_fit_keeps_only_the_last_checkpoint_when_asked(tmp_path):
    sold, logger = _fit(tmp_path, save_replay_buffer=False, keep_last_checkpoint_only=True)
    checkpoints_dir = Path(logger.log_dir) / "checkpoints"
    checkpoints = [p.name for p in checkpoints_dir.glob("*.ckpt")]
    assert len(checkpoints) == 1 and checkpoints[0].startswith("sold-steps=24-episode=3-")  # The one of step 0 is gone.
    assert sold.hparams["keep_last_checkpoint_only"] is True
    # The remaining checkpoint resumes.
    resumed = make_tiny_sold(max_steps=50)
    fit_sold(resumed, tmp_path / "second", checkpoint=checkpoints_dir / checkpoints[0])
    assert resumed.num_steps == 54


def test_delete_other_checkpoints_spares_the_kept_one_and_other_files(tiny_sold, tmp_path):
    tiny_sold.logger = SimpleNamespace(log_dir=str(tmp_path))
    checkpoints_dir = tiny_sold.checkpoints_dir
    for name in ["sold-steps=0-episode=0-eval_episode_return=1.0.ckpt", "sold-steps=24-episode=3.ckpt", "notes.txt"]:
        (checkpoints_dir / name).touch()
    (checkpoints_dir / "replay_buffer-steps=24-episode=3").mkdir()
    tiny_sold.delete_other_checkpoints(checkpoints_dir / "sold-steps=24-episode=3.ckpt")
    expected = ["notes.txt", "replay_buffer-steps=24-episode=3", "sold-steps=24-episode=3.ckpt"]
    assert sorted(os.listdir(checkpoints_dir)) == expected


def test_fit_keeps_only_the_latest_replay_buffer_copy(fitted):
    _, log_dir = fitted
    copies = sorted(p.name for p in (log_dir / "checkpoints").glob("replay_buffer-steps=*"))
    assert copies == ["replay_buffer-steps=24-episode=3"]
    copy_dir = log_dir / "checkpoints" / copies[0]
    assert sorted(os.listdir(copy_dir)) == ["action.dat", "bookkeeping.pt", "obs.dat", "reward.dat"]
    checkpoint = torch.load(sorted((log_dir / "checkpoints").glob("sold-steps=24*.ckpt"))[0], weights_only=False)
    assert checkpoint["num_steps"] == 24 and checkpoint["num_episodes"] == 3
    assert checkpoint["replay_buffer_path"] == copy_dir
    assert checkpoint["replay_buffer_capacity"] == 300 and checkpoint["replay_buffer_info"].num_episodes == 3
    assert set(checkpoint) >= {"state_dict", "optimizer_states", "hyper_parameters"}
    # The copied file describes the checkpoint's state, like the checkpoint keys.
    bookkeeping = RingBufferDataset.load_bookkeeping(copy_dir)
    assert bookkeeping["info"] == checkpoint["replay_buffer_info"] and bookkeeping["capacity"] == 300
    assert bookkeeping["episode_boundaries"] == checkpoint["replay_buffer_episode_boundaries"]


def test_fit_logs_metrics(fitted):
    _, log_dir = fitted
    records = read_metrics(log_dir)
    keys = set().union(*records)
    assert {
        "eval/episode_return",
        "eval/success_rate",
        "train/episode_return",
        "train/buffer_size",
        "train/dynamics_loss",
        "train/reward_loss",
        "train/actor_loss",
        "train/critic_loss",
        "gradients/actor",
    } <= keys
    timed = {
        *(f"collect/{name}" for name in ["action_selection", "env_step", "env_reset"]),
        *(f"replay_buffer/{name}" for name in ["add_step", "sample"]),
        *(f"learn/{name}" for name in ["autoencoder", "imagination", "training_step", "visualization", "logging"]),
        *(
            f"learn/{model}_{part}"
            for model in ["dynamics", "reward", "actor", "critic"]
            for part in ["loss", "update"]
        ),
        "learn/critic_target",
        *(f"eval/{name}" for name in ["action_selection", "env_step", "env_reset", "video_logging", "checkpoint"]),
        "eval/total",
    }
    assert {f"timing/{name}_ms" for name in timed} | {"timing/env_steps_per_second"} <= keys
    assert [r["step"] for r in records] == sorted(r["step"] for r in records)
    images = os.listdir(log_dir / "images")
    for name in [
        "autoencoder_reconstruction",
        "dynamics_prediction",
        "reward_prediction",
        "reward_predictor_attention",
        "latent_imagination",
        "actor_attention",
    ]:
        assert any(image.startswith(name + "-step=") for image in images), name


def test_fit_resumes_from_its_own_checkpoint(fitted, tmp_path, caplog):
    _, log_dir = fitted
    latest = sorted((log_dir / "checkpoints").glob("*.ckpt"))[-1]
    sold = make_tiny_sold(max_steps=50)
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        fit_sold(sold, tmp_path, checkpoint=latest)
    assert sold.num_steps == 54 and sold.num_episodes >= 3  # Rounds of 5 steps from step 24, until 50 is reached.
    assert sold.replay_buffer.num_episodes > 3  # The saved buffer (3 episodes) plus the new ones.
    copy_dir = log_dir / "checkpoints" / "replay_buffer-steps=24-episode=3"
    assert f"Restored 3 episodes (27 steps) of the replay buffer from {copy_dir}, described by" in caplog.text
    assert f"described by {copy_dir / 'bookkeeping.pt'}" in caplog.text


def test_fit_resumes_with_the_live_replay_buffer_without_a_copy(tmp_path, caplog):
    sold, logger = _fit(tmp_path / "first", save_replay_buffer=False)
    checkpoints_dir = Path(logger.log_dir) / "checkpoints"
    assert not list(checkpoints_dir.glob("replay_buffer-*"))
    latest = sorted(checkpoints_dir.glob("*.ckpt"))[-1]
    checkpoint = torch.load(latest, weights_only=False)
    live_dir = Path(logger.log_dir, "replay_buffer").resolve()
    assert checkpoint["replay_buffer_path"] == live_dir and checkpoint["replay_buffer_info"].num_episodes == 3
    assert checkpoint["replay_buffer_capacity"] == 300 and len(checkpoint["replay_buffer_episode_boundaries"]) == 3
    # The live file is newer than the checkpoint: it was rewritten before the fifth episode was stored.
    assert RingBufferDataset.load_bookkeeping(live_dir)["info"].num_episodes == 4

    resumed = make_tiny_sold(max_steps=50)
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        fit_sold(resumed, tmp_path / "second", checkpoint=latest)
    assert f"Restored 4 episodes (36 steps) of the replay buffer from {live_dir}" in caplog.text
    assert resumed.num_steps == 54 and resumed.replay_buffer.num_episodes == 4 + 3  # Episodes end at 32, 40 and 48.
    assert np.array_equal(resumed.replay_buffer.ring_buffer["obs"][:36], sold.replay_buffer.ring_buffer["obs"][:36])


def test_fit_resumes_with_a_given_replay_buffer_directory(tmp_path, caplog):
    # Checkpoints of older runs point to no buffer; replay_buffer= names one with its bookkeeping.pt.
    sold, logger = _fit(tmp_path / "first", save_replay_buffer=False)
    latest = sorted((Path(logger.log_dir) / "checkpoints").glob("*.ckpt"))[-1]
    checkpoint = torch.load(latest, weights_only=False)
    old = {key: value for key, value in checkpoint.items() if not key.startswith("replay_buffer_")}
    torch.save(old, tmp_path / "old.ckpt")
    live_dir = Path(logger.log_dir, "replay_buffer").resolve()

    resumed = make_tiny_sold(max_steps=50)
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        fit_sold(resumed, tmp_path / "second", checkpoint=tmp_path / "old.ckpt", replay_buffer=str(live_dir))
    assert f"Restored 4 episodes (36 steps) of the replay buffer from {live_dir}" in caplog.text
    assert np.array_equal(resumed.replay_buffer.ring_buffer["obs"][:36], sold.replay_buffer.ring_buffer["obs"][:36])

    (live_dir / "bookkeeping.pt").unlink()
    with pytest.raises(ValueError, match="rebuild the bookkeeping"):
        fit_sold(make_tiny_sold(), tmp_path / "third", checkpoint=tmp_path / "old.ckpt", replay_buffer=str(live_dir))
    with pytest.raises(ValueError, match="pass one"):
        fit_sold(make_tiny_sold(), tmp_path / "fourth", replay_buffer=str(live_dir))


def _optimizer_kernels(sold):
    return {(group["fused"], group["foreach"]) for optimizer in sold.optimizers for group in optimizer.param_groups}


def test_fit_uses_fused_optimizers_only_on_gpus_by_default(fitted):
    sold, _ = fitted
    assert not sold.fused_optimizers and _optimizer_kernels(sold) == {(None, None)}


def test_fused_and_unfused_optimizers_resume_from_each_others_checkpoints(fitted, tmp_path):
    _, log_dir = fitted
    latest = sorted((log_dir / "checkpoints").glob("*.ckpt"))[-1]
    saved = torch.load(latest, weights_only=False)["optimizer_states"]
    assert {group["fused"] for state in saved for group in state["param_groups"]} == {None}
    fused = make_tiny_sold(max_steps=30)
    fit_sold(fused, tmp_path / "fused", checkpoint=latest, fused_optimizers=True)
    assert fused.num_steps == 34 and _optimizer_kernels(fused) == {(True, None)}
    fused.save_checkpoint(tmp_path / "fused.ckpt")
    unfused = make_tiny_sold(max_steps=40)
    fit_sold(unfused, tmp_path / "unfused", checkpoint=tmp_path / "fused.ckpt")
    assert unfused.num_steps == 44 and _optimizer_kernels(unfused) == {(None, None)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_fit_on_a_gpu_uses_fused_optimizers(tmp_path):
    sold = make_tiny_sold(max_steps=20, save_replay_buffer=False)
    fit_sold(sold, tmp_path, device="cuda")
    assert sold.fused_optimizers and _optimizer_kernels(sold) == {(True, None)}


def test_fit_without_saving_replay_buffer_and_with_finetuning(tmp_path):
    sold, logger = _fit(
        tmp_path,
        save_replay_buffer=False,
        finetune_autoencoder=True,
        max_steps=20,
        start_imagination_from_every=True,
        num_context=(1, 2),
    )
    assert sold.num_steps == 20
    assert not list(Path(logger.log_dir, "checkpoints").glob("replay_buffer-*"))
    assert "timing/learn/autoencoder_update_ms" in set().union(*read_metrics(logger.log_dir))
    checkpoint = torch.load(next(Path(logger.log_dir, "checkpoints").glob("*.ckpt")), weights_only=False)
    assert checkpoint["num_steps"] == 0 and checkpoint["replay_buffer_info"].is_empty  # Saved before any episode.
    assert checkpoint["replay_buffer_path"] == Path(logger.log_dir, "replay_buffer").resolve()


def test_fit_without_profiling_logs_no_timings(tmp_path):
    _, logger = _fit(tmp_path, max_steps=20, profile=False)
    records = read_metrics(logger.log_dir)
    assert records and not any(key.startswith("timing/") for key in set().union(*records))


# ----------------------------------------------------------------------------- checkpoints
def test_load_counters(tiny_sold):
    tiny_sold.load_counters({"num_steps": 300, "num_episodes": 12})
    assert (tiny_sold.num_steps, tiny_sold.num_episodes) == (300, 11)


def _resume_checkpoint(tiny_sold, **extra):
    return {"state_dict": tiny_sold.state_dict(), "optimizer_states": [], "num_steps": 5, "num_episodes": 1, **extra}


def test_restore_rejects_too_many_optimizer_states(tiny_sold):
    tiny_sold.optimizers = tiny_sold.configure_optimizers()
    states = [optimizer.state_dict() for optimizer in tiny_sold.optimizers] * 2
    with pytest.raises(ValueError, match="8 optimizer states for 4 optimizers"):
        tiny_sold.restore(_resume_checkpoint(tiny_sold, optimizer_states=states))


def test_restore_starts_missing_optimizers_fresh_with_a_warning(tiny_sold):
    # Checkpoints of fine-tuning runs saved before BUGS_AND_PITFALLS.md #17 was fixed lack the autoencoder's state.
    tiny_sold.optimizers = tiny_sold.configure_optimizers()
    for optimizer in tiny_sold.optimizers:
        for group in optimizer.param_groups:
            group["lr"] = 0.5
    states = [optimizer.state_dict() for optimizer in tiny_sold.optimizers[:3]]
    for optimizer in tiny_sold.optimizers:
        for group in optimizer.param_groups:
            group["lr"] = 0.25
    assert tiny_sold.num_updates_since_restore is None
    with pytest.warns(UserWarning, match="3 optimizer states for 4 optimizers: the last 1 start fresh"):
        tiny_sold.restore(_resume_checkpoint(tiny_sold, optimizer_states=states))
    assert tiny_sold.num_updates_since_restore == 0
    # The run's learning rate wins over the checkpoint's; a fresh optimizer has no state.
    assert all(group["lr"] == 0.25 for optimizer in tiny_sold.optimizers for group in optimizer.param_groups)
    assert tiny_sold.optimizers[-1].state_dict()["state"] == {}


def test_restore_without_replay_buffer(tiny_sold, tmp_path):
    tiny_sold.restore(_resume_checkpoint(tiny_sold))
    assert (tiny_sold.num_steps, tiny_sold.num_episodes) == (5, 0)
    with pytest.warns(UserWarning, match="no longer exists: resuming with an empty one"):
        tiny_sold.restore(_resume_checkpoint(tiny_sold, replay_buffer_path=tmp_path / "gone"))
    assert tiny_sold.replay_buffer is None


def _saved_buffer(tmp_path, num_episodes=1, capacity=20):
    """A buffer at tmp_path/saved holding num_episodes episodes of 5 steps, its bookkeeping.pt up to date."""
    saved = RingBufferDataset(capacity, 2, 3, save_path=str(tmp_path / "saved"))
    for index in range(num_episodes):
        add_episode(saved, make_episode(5, start=10 * index))
    saved.save_bookkeeping()
    return saved


def _buffer_checkpoint(tiny_sold, saved, **extra):
    return _resume_checkpoint(
        tiny_sold,
        replay_buffer_path=Path(saved.save_path),
        replay_buffer_capacity=saved.capacity,
        replay_buffer_info=copy.deepcopy(saved.info),
        replay_buffer_head=saved.head,
        replay_buffer_tail=saved.tail,
        replay_buffer_episode_boundaries=deque(saved.episode_boundaries),
        **extra,
    )


def test_restore_rejects_buffer_without_fields(tiny_sold, tmp_path):
    from sold.data.info import EpisodeDatasetInfo

    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    checkpoint = _resume_checkpoint(
        tiny_sold,
        replay_buffer_path=tmp_path,
        replay_buffer_capacity=20,
        replay_buffer_info=EpisodeDatasetInfo(num_episodes=1),
        replay_buffer_head=0,
        replay_buffer_tail=0,
        replay_buffer_episode_boundaries=deque(),
    )
    with pytest.raises(ValueError, match="without fields"):
        tiny_sold.restore(checkpoint)


def test_restore_rejects_a_different_buffer_capacity(tiny_sold, tmp_path):
    tiny_sold.replay_buffer = RingBufferDataset(30, 2, 3, save_path=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="holds 20 steps, but this run's buffer_capacity is 30"):
        tiny_sold.restore(_buffer_checkpoint(tiny_sold, _saved_buffer(tmp_path)))


def test_restore_skips_an_empty_saved_buffer(tiny_sold, tmp_path):
    saved = _saved_buffer(tmp_path, num_episodes=0)  # As the checkpoint of step 0 describes it.
    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    tiny_sold.restore(_buffer_checkpoint(tiny_sold, saved))
    assert tiny_sold.replay_buffer.is_empty and tiny_sold.replay_buffer.fields is None


# BUGS_AND_PITFALLS.md #4 (fixed).
def test_resuming_restores_the_saved_replay_buffer(tiny_sold, tmp_path, caplog):
    saved = _saved_buffer(tmp_path)
    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        tiny_sold.restore(_buffer_checkpoint(tiny_sold, saved))
    assert tiny_sold.replay_buffer.num_episodes == 1 and tiny_sold.replay_buffer.tail == 5
    assert np.array_equal(tiny_sold.replay_buffer.ring_buffer["obs"], saved.ring_buffer["obs"])
    assert f"Restored 1 episodes (5 steps) of the replay buffer from {saved.save_path}, described by" in caplog.text


def test_restore_prefers_the_buffers_bookkeeping_file_to_the_checkpoint(tiny_sold, tmp_path, caplog):
    saved = _saved_buffer(tmp_path)
    checkpoint = _buffer_checkpoint(tiny_sold, saved)  # Describes one episode; the run then stores a second one.
    add_episode(saved, make_episode(5, start=10))
    saved.save_bookkeeping()
    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        tiny_sold.restore(checkpoint)
    assert tiny_sold.replay_buffer.num_episodes == 2 and tiny_sold.replay_buffer.tail == 10
    assert f"described by {saved.save_path}/bookkeeping.pt" in caplog.text


def test_restore_falls_back_to_the_checkpoints_bookkeeping(tiny_sold, tmp_path, caplog):
    """Checkpoints from before bookkeeping.pt existed still resume."""
    saved = _saved_buffer(tmp_path)
    checkpoint = _buffer_checkpoint(tiny_sold, saved)
    add_episode(saved, make_episode(5, start=10))
    os.remove(Path(saved.save_path, "bookkeeping.pt"))
    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    with caplog.at_level(logging.INFO, logger="sold.utils.training"):
        tiny_sold.restore(checkpoint)
    assert tiny_sold.replay_buffer.num_episodes == 1 and tiny_sold.replay_buffer.tail == 5
    assert "described by the checkpoint" in caplog.text


def test_replay_buffer_checkpoint_points_to_the_live_buffer(tiny_sold, tmp_path):
    tiny_sold.logger = SimpleNamespace(log_dir=str(tmp_path))
    tiny_sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "live"))  # Empty, no files yet.
    checkpoint = tiny_sold.replay_buffer_checkpoint()
    assert checkpoint["replay_buffer_path"] == (tmp_path / "live").resolve()
    assert os.listdir(tmp_path / "live") == ["bookkeeping.pt"] and not (tmp_path / "checkpoints").exists()
    assert checkpoint["replay_buffer_capacity"] == 20 and checkpoint["replay_buffer_info"].is_empty
    assert RingBufferDataset.load_bookkeeping(checkpoint["replay_buffer_path"])["info"].is_empty


def test_checkpoints_dir_is_created_under_log_dir(tiny_sold, tmp_path):
    tiny_sold.logger = SimpleNamespace(log_dir=str(tmp_path))
    assert tiny_sold.checkpoints_dir == tmp_path / "checkpoints" and (tmp_path / "checkpoints").is_dir()


# BUGS_AND_PITFALLS.md #5 (fixed): the previous copy is removed even where text order would put it last.
def test_replay_buffer_checkpoint_replaces_the_previous_copy(tiny_sold, tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "obs.dat").write_bytes(b"new")
    (tmp_path / "checkpoints" / "replay_buffer-steps=9000-episode=180").mkdir(parents=True)
    tiny_sold.logger = SimpleNamespace(log_dir=str(tmp_path))
    tiny_sold.replay_buffer = SimpleNamespace(
        save_path=str(live),
        capacity=1,
        info=None,
        head=0,
        tail=0,
        episode_boundaries=[],
        save_bookkeeping=lambda: None,
    )
    tiny_sold.save_replay_buffer = True
    tiny_sold.num_steps, tiny_sold.num_episodes = 10500, 210
    checkpoint = tiny_sold.replay_buffer_checkpoint()
    assert [p.name for p in (tmp_path / "checkpoints").iterdir()] == ["replay_buffer-steps=10500-episode=210"]
    assert (checkpoint["replay_buffer_path"] / "obs.dat").read_bytes() == b"new"
    assert checkpoint["replay_buffer_capacity"] == 1


# BUGS_AND_PITFALLS.md #6 (fixed).
def test_latest_checkpoint_loads_without_training_state(fitted):
    from sold.train_sold import SOLDModule

    _, log_dir = fitted
    checkpoints_dir = log_dir / "checkpoints"
    assert list(checkpoints_dir.glob("replay_buffer-*")), "precondition: the replay buffer copy exists"
    latest = sorted(checkpoints_dir.glob("*.ckpt"))[-1]
    loaded = SOLDModule.load(latest, env=fakes.make_fake_env(), max_steps=99)  # As evaluate_sold.py does.
    assert loaded.num_steps == torch.load(latest, weights_only=False)["num_steps"]
    assert loaded.max_steps == 99 and loaded.replay_buffer is None and loaded.optimizers == []
