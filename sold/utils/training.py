from abc import ABC, abstractmethod
import gym
import hydra
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
import numpy as np
import os
from pathlib import Path
import random
from rich.progress import TaskID
import shutil
from sold.data.ring_buffer import RingBufferDataset
from sold.data.utils import NumUpdatesWrapper
from sold.utils.episode_runner import EpisodeRunner
from sold.utils.logging import LoggingStepMixin
from sold.utils.profiling import Timings
from termcolor import colored
import time
import torch
from torch.utils.data import DataLoader
from typing import Any
import warnings


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def print_summary(cfg):
    color, attrs = "magenta", ["bold"]

    def _pprint(k, v):
        print(colored(f"{k + ':':<15}", color, attrs=attrs), v)

    kvs = [
        ("Output Dir", hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
        ("Task", cfg.model.env.name),
        ("Max steps", f"{int(cfg.model.max_steps):,}"),
        ("Experiment", cfg.experiment),
    ]
    for k, v in kvs:
        _pprint(k, v)


class OnlineProgressBar(RichProgressBar):
    """Custom progress bar for online RL training loops."""

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Create the display and the single train task up front: Lightning >= 2.5.3 adds the task itself in
        # on_train_batch_end if none exists, which fails when the display does not exist yet.
        self.pl_module = pl_module
        super().on_train_start(trainer, pl_module)
        if self.progress is not None:
            self.train_progress_bar_id = self._add_task(
                self.total_train_batches, self._get_train_description(trainer.current_epoch)
            )

    def _get_train_description(self, current_epoch: int) -> str:
        train_description = f"Steps {self.pl_module.num_steps}, Episodes {max(self.pl_module.num_episodes, 0)}"
        return train_description

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return

    def _update(self, progress_bar_id: TaskID | None, current: int, visible: bool = True) -> None:
        if self.progress is not None and self.is_enabled:
            self.progress.update(
                self.train_progress_bar_id,
                completed=self.pl_module.num_steps,
                visible=visible,
                description=self._get_train_description(self.trainer.current_epoch),
            )
            self.refresh()

    @property
    def total_train_batches(self) -> int | float:
        return self.pl_module.max_steps


class OnlineModule(LoggingStepMixin, LightningModule, ABC):
    def __init__(
        self,
        env: gym.Env,
        max_steps: int,
        num_seed: int,
        update_freq: int,
        num_updates: int,
        eval_freq: int,
        num_eval_episodes: int,
        batch_size: int,
        sequence_length: int,
        buffer_capacity: int,
        save_replay_buffer: bool,
        profile: bool = True,
    ) -> None:
        """Integrates online experience collection with PyTorch Lightning's training loop.

        Args:
            env (gym.Env): The environment to interact with.
            max_steps (int): Maximum number of environment steps to collect before stopping training.
            num_seed (int): Number of seed steps/episodes to collect before training.
            update_freq (int): Update the agent every 'update_freq' environment steps.
            num_updates (int): Number of updates to perform whenever the agent is being updated.
            eval_freq (int): Evaluate the agent every 'eval_freq' environment steps.
            num_eval_episodes (int): Number of episodes to collect when evaluating the agent.
            batch_size (int): Batch size of experience sampled from the replay buffer.
            sequence_length (int): Length of sequences sampled from the replay buffer.
            buffer_capacity (int): Maximum number of steps to store in the replay buffer.
            save_replay_buffer (bool): Save the replay buffer with each checkpoint.
            profile (bool): Log section durations as timing/*_ms.
        """
        super().__init__()
        self.env = env
        self.max_steps = max_steps
        self.num_seed = num_seed
        self.update_freq = update_freq
        self.num_updates = num_updates
        self.eval_freq = eval_freq
        self.num_eval_episodes = num_eval_episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.buffer_capacity = buffer_capacity
        self.save_replay_buffer = save_replay_buffer

        self.last_action = torch.full_like(
            torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float("nan")
        ).to(self.device)
        self.timings = Timings(enabled=profile)
        self.runner = EpisodeRunner(env, self._policy, self.timings)
        self.num_steps = 0
        self.num_episodes = -1  # Start at -1 to account for the initial reset.

    @property
    def checkpoints_dir(self) -> Path:
        checkpoints_dir = Path(self.logger.log_dir) / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        pass

    def measure_gpu(self, name: str):
        """Time a section that queues GPU work (CUDA events on a GPU)."""
        return self.timings.measure(name, cuda=self.device.type == "cuda")

    def _policy(self, obs: torch.Tensor, is_first: bool, mode: str) -> torch.Tensor:
        # select_action reads last_action to encode the next observation.
        self.last_action[:] = self.select_action(obs.to(self.device), is_first=is_first, mode=mode)
        return self.last_action

    def on_fit_start(self) -> None:
        self.eval_next = False
        self.after_eval = False
        self.runner.done = True  # Start with a new episode.
        self.last_timings_log = (time.perf_counter(), self.num_steps)

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "replay_buffer"):
            self.replay_buffer = RingBufferDataset(
                self.buffer_capacity,
                self.batch_size,
                self.sequence_length,
                save_path=self.logger.log_dir + "/replay_buffer",
            )
        dataset = NumUpdatesWrapper(self.replay_buffer, self.num_updates, measure_sampling=self.timings.enabled)
        # The replay buffer is sampled in a single worker process on purpose.
        warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=1)
        return dataloader

    @torch.no_grad()
    def on_train_epoch_start(self) -> None:
        num_collect_steps = self.num_seed if self.num_steps == 0 and self.num_seed > 0 else self.update_freq
        for _ in range(num_collect_steps):
            self.collect_step()
        self.batch_mark = time.perf_counter()

    @torch.no_grad()
    def collect_step(self) -> None:
        if self.num_steps % self.eval_freq == 0:
            self.eval_next = True

        if self.runner.done:
            self.num_episodes += 1
            if self.eval_next:
                with self.timings.measure("eval/total"):
                    self.eval_loop()

            # Reset environment and store initial observation.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.replay_buffer.last_episode_return is not None:
                self.log("train/episode_return", self.replay_buffer.last_episode_return.item(), prog_bar=True)
            obs = self.runner.reset()
            with self.timings.measure("replay_buffer/add_step"):
                self.replay_buffer.add_step(self._complete_first_timestep({"obs": obs}))
            self.log("train/num_episodes", self.num_episodes)
            self.log("train/num_steps", self.num_steps)

        # Select action, perform environment step, and store resulting experience.
        mode = "train" if self.num_steps >= self.num_seed else "random"
        obs, reward, done, info = self.runner.step(mode)
        with self.timings.measure("replay_buffer/add_step"):
            self.replay_buffer.add_step({"obs": obs, "action": self.last_action.cpu(), "reward": reward}, done=done)
        self.num_steps += 1

    def _complete_first_timestep(self, step_data: dict[str, Any]) -> dict[str, Any]:
        if "action" not in step_data:
            step_data["action"] = torch.full_like(
                torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float("nan")
            )
        if "reward" not in step_data:
            step_data["reward"] = torch.tensor(float("nan"))
        return step_data

    @torch.no_grad()
    def eval_loop(self) -> None:
        episode_returns, successes = [], []
        for episode_index in range(self.num_eval_episodes):
            episode = self.runner.play_episode(mode="eval")
            with self.timings.measure("eval/video_logging"):
                self.log_video(f"eval/episode_{episode_index}", torch.stack(episode["obs"]))
            episode_returns.append(sum(episode["reward"]))
            if "success" in episode:
                successes.append(episode["success"])
        self.log("eval/episode_return", np.mean(episode_returns), prog_bar=True)
        if successes:
            self.log("eval/success_rate", np.mean(successes))

        for episode_index in range(3):
            episode = self.runner.play_episode(mode="train")
            with self.timings.measure("eval/video_logging"):
                self.log_video(f"train/episode_{episode_index}", torch.stack(episode["obs"]))

        with self.timings.measure("eval/checkpoint"):
            self.trainer.save_checkpoint(
                self.checkpoints_dir / f"sold-steps={self.num_steps}-episode={self.num_episodes}-"
                f"eval_episode_return={np.mean(episode_returns)}.ckpt"
            )

        self.eval_next = False
        self.after_eval = True

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if "sample_seconds" in batch:  # From the DataLoader worker.
            for seconds in batch.pop("sample_seconds").tolist():
                self.timings.add("replay_buffer/sample", seconds)
        return batch

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        self._time_since_batch_mark("replay_buffer/batch_wait")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.after_eval = False  # Reset 'after_eval' to False after the first training batch.
        self._time_since_batch_mark("learn/training_step")

    def _time_since_batch_mark(self, name: str) -> None:
        now = time.perf_counter()
        self.timings.add(name, now - self.batch_mark)
        self.batch_mark = now

    def on_train_epoch_end(self) -> None:
        if self.timings.enabled and self.device.type == "cuda":
            with self.timings.measure("learn/gpu_drain"):
                torch.cuda.synchronize(self.device)
        self.log_timings()
        if self.num_steps >= self.max_steps:
            self.trainer.should_stop = True

    def log_timings(self) -> None:
        """Log the section timings and env steps per second since the last call."""
        if not (self.timings.enabled and self.is_logging_step("timing/env_steps_per_second")):
            return
        for name, milliseconds in self.timings.summary().items():
            self.log(f"timing/{name}_ms", milliseconds)
        last_time, last_num_steps = self.last_timings_log
        now = time.perf_counter()
        self.log("timing/env_steps_per_second", (self.num_steps - last_num_steps) / (now - last_time))
        self.last_timings_log = (now, self.num_steps)

    @property
    def logging_step(self) -> int:
        return self.num_steps

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["num_steps"] = self.num_steps
        checkpoint["num_episodes"] = self.num_episodes

        # Save the state of the replay buffer for the last checkpoint to enable checkpoint/restart (CPR) functionality.
        if self.save_replay_buffer:
            replay_buffer_dir = (
                self.checkpoints_dir / f"replay_buffer-steps={self.num_steps}-episode={self.num_episodes}"
            )
            if Path(self.replay_buffer.save_path).resolve().exists():
                shutil.copytree(Path(self.replay_buffer.save_path).resolve(), replay_buffer_dir)
            # Keep only the copy for this checkpoint (sorting the names as text would misorder 9000 and 10500).
            for old_replay_buffer_dir in self.checkpoints_dir.glob("replay_buffer-steps=*"):
                if old_replay_buffer_dir != replay_buffer_dir:
                    shutil.rmtree(old_replay_buffer_dir)

            checkpoint["replay_buffer_path"] = replay_buffer_dir
            checkpoint["replay_buffer_capacity"] = self.replay_buffer.capacity
            checkpoint["replay_buffer_info"] = self.replay_buffer.info
            checkpoint["replay_buffer_head"] = self.replay_buffer.head
            checkpoint["replay_buffer_tail"] = self.replay_buffer.tail
            checkpoint["replay_buffer_episode_boundaries"] = self.replay_buffer.episode_boundaries

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.num_steps = checkpoint["num_steps"]
        self.num_episodes = checkpoint["num_episodes"] - 1  # Incremented at once, since self.runner.done is True.

        # The replay buffer is only needed to resume training. Without a trainer (e.g. load_from_checkpoint in
        # evaluate_sold.py) there is no logger directory to restore it into.
        replay_buffer_path = Path(checkpoint["replay_buffer_path"]) if "replay_buffer_path" in checkpoint else None
        if replay_buffer_path is not None and replay_buffer_path.exists() and self._trainer is not None:
            self.train_dataloader()  # Initializes the replay buffer.
            if checkpoint["replay_buffer_info"].fields is None:
                raise ValueError("Cannot load replay buffer without fields.")
            self.replay_buffer.load_from_disk_storage(
                checkpoint["replay_buffer_path"],
                checkpoint["replay_buffer_capacity"],
                checkpoint["replay_buffer_info"],
                checkpoint["replay_buffer_head"],
                checkpoint["replay_buffer_tail"],
                checkpoint["replay_buffer_episode_boundaries"],
            )
