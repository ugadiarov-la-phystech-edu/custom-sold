from abc import ABC, abstractmethod
from collections.abc import Sequence
import gym
import hydra
import logging
import numpy as np
import os
from pathlib import Path
import random
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
import shutil
from sold.data.ring_buffer import BOOKKEEPING_FILE, RingBufferDataset
from sold.data.utils import sample_batches
from sold.envs.vector import SingleEnv
from sold.utils.episode_runner import EpisodeRunner
from sold.utils.logging import MetricsWriter
from sold.utils.profiling import Timings
import sys
from termcolor import colored
import time
import torch
from torch import nn
from torch.optim import Optimizer
from typing import Any
import warnings

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def resolve_device(device: str | torch.device) -> torch.device:
    """ "auto" is the GPU when there is one."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def _autocast(method, device_type: str):
    """method run under bfloat16 autocast, with its floating-point tensor outputs (also in dicts) cast to float32."""

    def to_float32(output):
        if isinstance(output, dict):
            return {key: to_float32(value) for key, value in output.items()}
        return output.float() if output.is_floating_point() else output

    def autocast_method(*args, **kwargs):
        with torch.autocast(device_type, dtype=torch.bfloat16):
            output = method(*args, **kwargs)
        return to_float32(output)

    return autocast_method


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


class ProgressBar:
    """Progress over environment steps, shown when stdout is a terminal."""

    def __init__(self, total: int, enabled: bool | None = None) -> None:
        enabled = sys.stdout.isatty() if enabled is None else enabled
        columns = [TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()]
        self.progress = Progress(*columns, TimeRemainingColumn(), disable=not enabled)
        self.task = self.progress.add_task("", total=total)
        self.eval_return: float | None = None

    def __enter__(self) -> "ProgressBar":
        self.progress.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.progress.stop()

    def update(self, num_steps: int, num_episodes: int) -> None:
        description = f"Steps {num_steps}, Episodes {max(num_episodes, 0)}"
        if self.eval_return is not None:
            description += f", eval return {self.eval_return:.2f}"
        self.progress.update(self.task, completed=num_steps, description=description)


class OnlineModule(nn.Module, ABC):
    """Online RL training: collect experience with the policy, update from the replay buffer, evaluate, checkpoint."""

    GRADIENT_NORMS: tuple[str, ...] = ()  # Submodules whose gradient norms are logged after each round of updates.
    STAGES: tuple[str, ...] = ()  # Training stages compile_stages can compile, as methods _<stage>_stage.

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
        keep_last_checkpoint_only: bool = False,
    ) -> None:
        """
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
            save_replay_buffer (bool): Copy the replay buffer's files next to each checkpoint (see restore).
            profile (bool): Log section durations as timing/*_ms.
            keep_last_checkpoint_only (bool): Delete the run's earlier checkpoints whenever a new one is saved.
        """
        super().__init__()
        self.env = env if hasattr(env, "num_envs") else SingleEnv(env)  # A batch of environments (sold.envs.vector).
        self.num_envs = self.env.num_envs
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
        self.keep_last_checkpoint_only = keep_last_checkpoint_only
        self.hparams: dict[str, Any] = {}  # Constructor arguments, saved in checkpoints to rebuild the module.

        self.register_buffer("last_action", self._nan_actions(), persistent=False)  # (num_envs, action_dim)
        self.timings = Timings(enabled=profile)
        self.runner = EpisodeRunner(self.env, self._policy, self.timings)
        self.num_steps = 0
        self.num_episodes = -self.num_envs  # The initial reset adds num_envs.
        self.update_carry = 0  # Remainder of num_updates / update_freq updates owed per env step (update_count).
        self.finished_returns: list[torch.Tensor] = []  # Returns of the episodes finished since the last reset.
        # Set up by fit().
        self.optimizers: list[Optimizer] = []
        self.num_updates_since_restore: int | None = None  # Counts gradient updates after resuming (None: fresh run).
        self.learning_rates: dict[str, float] = {}  # Learning rates subclasses report (e.g. during a warmup).
        self.fused_optimizers = False  # Fused Adam kernels (configure_optimizers reads it).
        self.cudagraphs = False  # Compiled stages replay CUDA graphs.
        self.logger: MetricsWriter | None = None
        self.replay_buffer: RingBufferDataset | None = None
        self.progress = ProgressBar(max_steps)
        self.last_timings_log = (time.perf_counter(), 0)
        self.eval_next = False
        self.after_eval = False  # Visualize the next update (set by eval_loop).

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def checkpoints_dir(self) -> Path:
        checkpoints_dir = Path(self.logger.log_dir) / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        return checkpoints_dir

    @abstractmethod
    def select_action(self, obs: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        """The actions (num_envs, action_dim) after observing obs (num_envs, ...), in mode "train" (sampled), "eval"
        (mode) or "random"."""

    @abstractmethod
    def training_step(self, batch: dict[str, torch.Tensor], visualize: bool = False) -> dict[str, Any]:
        """One gradient update on a batch (with figures if visualize); returns the outputs, losses under *_loss."""

    @abstractmethod
    def configure_optimizers(self) -> list[Optimizer]:
        """The optimizers whose states checkpoints store."""

    # ---------------------------------------------------------------------------------------------- training loop
    def fit(
        self,
        logger: MetricsWriter,
        device: str | torch.device = "auto",
        checkpoint: str | None = None,
        replay_buffer: str | None = None,
        fused_optimizers: bool | None = None,
        compile_mode: str | None = None,
        compile_stages: Sequence[str] | None = None,
        autocast_dtype: str | None = None,
        autocast_stages: Sequence[str] | None = None,
    ) -> None:
        """Train for max_steps environment steps, logging to logger; checkpoint resumes a saved run.

        replay_buffer: a buffer directory (with its bookkeeping.pt) to resume with instead of the one the checkpoint
        points to, e.g. for checkpoints of older runs (see scripts/rebuild_replay_buffer_bookkeeping.py).
        fused_optimizers: fused optimizer kernels; None uses them on GPUs. compile_mode: torch.compile mode of the
        compile_stages (all STAGES if None); None trains eagerly. autocast_dtype: "bfloat16" runs the autocast_stages
        (all STAGES if None) in mixed precision (see autocast_stages); None trains in float32.
        """
        if replay_buffer is not None and checkpoint is None:
            raise ValueError("replay_buffer resumes a checkpoint: pass one.")
        self.to(resolve_device(device))
        self.fused_optimizers = self.device.type == "cuda" if fused_optimizers is None else fused_optimizers
        self.train()
        self.logger = logger
        self.replay_buffer = RingBufferDataset(
            self.buffer_capacity, self.batch_size, self.sequence_length, save_path=logger.log_dir + "/replay_buffer"
        )
        self.optimizers = self.configure_optimizers()
        if checkpoint is not None:
            saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
            if replay_buffer is not None:
                saved["replay_buffer_path"] = replay_buffer
            self.restore(saved)
        if compile_mode is not None:  # After moving and restoring: CUDA graphs keep the parameters' memory.
            self.compile_stages(compile_mode, compile_stages)
        if autocast_dtype is not None:
            self.autocast_stages(autocast_dtype, autocast_stages)
        self.runner.done = True  # Start with a new episode.
        self.last_timings_log = (time.perf_counter(), self.num_steps)

        with self.progress:
            if self.num_steps == 0:
                # Draws the sampler's seed before the first collection, as the old loop did; run RNG streams rely on it.
                self.update()
            self.collect(self.num_seed if self.num_steps == 0 and self.num_seed > 0 else self.update_freq)
            while self.num_steps < self.max_steps:
                self.update(self.update_count(self.collect(self.update_freq)))
        logger.finalize()

    def update_count(self, num_steps: int) -> int:
        """How many updates num_steps new env steps earn: num_updates per update_freq steps, carrying the remainder."""
        num_updates, self.update_carry = divmod(num_steps * self.num_updates + self.update_carry, self.update_freq)
        return num_updates

    @torch.no_grad()
    def collect(self, num_steps: int) -> int:
        """Collect at least num_steps env steps (whole batches of num_envs); return how many."""
        start = self.num_steps
        while self.num_steps < start + num_steps:
            self.collect_step()
            self.progress.update(self.num_steps, self.num_episodes)
        return self.num_steps - start

    @torch.no_grad()
    def collect_step(self) -> None:
        """One step of every environment, stored in the replay buffer (each environment's episode separately)."""
        next_eval_step = -(-self.num_steps // self.eval_freq) * self.eval_freq  # The first multiple from num_steps.
        if next_eval_step < self.num_steps + self.num_envs:
            self.eval_next = True

        if self.runner.done:
            self.num_episodes += self.num_envs
            if self.eval_next:
                with self.timings.measure("eval/total"):
                    self.eval_loop()

            # Reset the environments and store the initial observations.
            self.log("train/buffer_size", self.replay_buffer.num_timesteps)
            if self.finished_returns:
                self.log("train/episode_return", torch.stack(self.finished_returns).mean())
                self.finished_returns = []
            obs = self.runner.reset()
            with self.timings.measure("replay_buffer/add_step"):
                for env_index, env_obs in enumerate(obs):
                    first = {"obs": env_obs, "action": self._nan_action(), "reward": torch.tensor(float("nan"))}
                    self.replay_buffer.add_step(first, stream=env_index)
            self.log("train/num_episodes", self.num_episodes)
            self.log("train/num_steps", self.num_steps)

        # Select actions, step the environments, and store the resulting experience.
        mode = "train" if self.num_steps >= self.num_seed else "random"
        obs, rewards, dones, _ = self.runner.step(mode)
        with self.timings.measure("replay_buffer/add_step"):
            for env_index, (env_obs, action, reward, done) in enumerate(zip(obs, self.runner.action, rewards, dones)):
                step = {"obs": env_obs, "action": action, "reward": reward}
                self.replay_buffer.add_step(step, done=done, stream=env_index)
                if done:
                    self.finished_returns.append(self.replay_buffer.last_episode_return)
        self.num_steps += self.num_envs

    def update(self, num_updates: int | None = None) -> None:
        """A round of gradient updates on replay-buffer batches (num_updates of them), then the round's metrics."""
        outputs, device = None, self.device
        visualize, self.after_eval = self.after_eval, False
        num_updates = self.num_updates if num_updates is None else num_updates
        for index, batch in enumerate(sample_batches(self.replay_buffer, num_updates, self.timings)):
            if self.cudagraphs:  # The previous update's graph outputs may be overwritten from here on.
                torch.compiler.cudagraph_mark_step_begin()
            with self.timings.measure("learn/training_step"):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = self.training_step(batch, visualize=visualize and index == 0)
            if self.num_updates_since_restore is not None:
                self.num_updates_since_restore += 1
        if outputs is not None:
            self.log_round(outputs)

    def compile_stages(self, mode: str, stages: Sequence[str] | None = None) -> None:
        """Replace the _<stage>_stage methods of stages (all STAGES if None) by their torch.compile'd versions."""
        for stage in self._stage_names(stages):
            method = getattr(self, f"_{stage}_stage")
            # fullgraph: a graph break raises instead of silently running eagerly.
            setattr(self, f"_{stage}_stage", torch.compile(method, mode=mode, fullgraph=True, dynamic=False))
        self.cudagraphs = mode in ("reduce-overhead", "max-autotune")

    def autocast_stages(self, dtype: str, stages: Sequence[str] | None = None) -> None:
        """Run the _<stage>_stage methods of stages (all STAGES if None) under autocast to dtype ("bfloat16").

        Autocast runs matmuls, convolutions and attention in bfloat16; parameters, gradients and optimizer states stay
        float32, and the stages' floating-point outputs are cast back to float32. Call it on the final device and after
        compile_stages: compiled code is traced with the autocast around it (and guarded on it), whereas compiling the
        wrappers would cache every stage under the wrapper's one code object.
        """
        if dtype != "bfloat16":  # float16 would need loss scaling.
            raise ValueError(f"Unsupported autocast dtype {dtype!r}; expected 'bfloat16'.")
        for stage in self._stage_names(stages):
            setattr(self, f"_{stage}_stage", _autocast(getattr(self, f"_{stage}_stage"), self.device.type))

    def _stage_names(self, stages: Sequence[str] | None) -> tuple[str, ...]:
        stages = self.STAGES if stages is None else tuple(stages)
        if unknown := set(stages) - set(self.STAGES):
            raise ValueError(f"Unknown stages {sorted(unknown)}; expected some of {self.STAGES}.")
        return stages

    def _policy(self, obs: torch.Tensor, is_first: bool, mode: str) -> torch.Tensor:
        if self.cudagraphs:  # Outputs of earlier replays are copied out by now.
            torch.compiler.cudagraph_mark_step_begin()
        # select_action reads last_action to encode the next observation.
        self.last_action[:] = self.select_action(obs.to(self.device), is_first=is_first, mode=mode)
        return self.last_action

    def _nan_action(self) -> torch.Tensor:
        return torch.full_like(torch.from_numpy(self.env.action_space.sample().astype(np.float32)), float("nan"))

    def _nan_actions(self) -> torch.Tensor:
        return torch.stack([self._nan_action() for _ in range(self.num_envs)])

    @torch.no_grad()
    def eval_loop(self) -> None:
        episode_returns, successes = [], []
        for episode_index, episode in enumerate(self.runner.play_episodes("eval", self.num_eval_episodes)):
            with self.timings.measure("eval/video_logging"):
                self.log_video(f"eval/episode_{episode_index}", torch.stack(episode["obs"]))
            episode_returns.append(sum(episode["reward"]))
            if "success" in episode:
                successes.append(episode["success"])
        self.log("eval/episode_return", np.mean(episode_returns))
        self.progress.eval_return = float(np.mean(episode_returns))
        if successes:
            self.log("eval/success_rate", np.mean(successes))

        for episode_index, episode in enumerate(self.runner.play_episodes("train", 3)):
            with self.timings.measure("eval/video_logging"):
                self.log_video(f"train/episode_{episode_index}", torch.stack(episode["obs"]))

        with self.timings.measure("eval/checkpoint"):
            path = self.checkpoints_dir / (
                f"sold-steps={self.num_steps}-episode={self.num_episodes}-"
                f"eval_episode_return={np.mean(episode_returns)}.ckpt"
            )
            self.save_checkpoint(path)
            if self.keep_last_checkpoint_only:
                self.delete_other_checkpoints(path)

        self.eval_next = False
        self.after_eval = True

    # ---------------------------------------------------------------------------------------------------- logging
    def log(self, name: str, value: torch.Tensor | float) -> None:
        self.logger.log_metrics({name: value}, step=self.num_steps)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        self.logger.log_image(name, image, step=self.num_steps)

    def log_video(self, name: str, video: torch.Tensor) -> None:
        self.logger.log_video(name, video, step=self.num_steps)

    def measure_gpu(self, name: str):
        """Time a section that queues GPU work (CUDA events on a GPU)."""
        return self.timings.measure(name, cuda=self.device.type == "cuda")

    def gradient_norm(self, model_name: str) -> torch.Tensor:
        model = getattr(self, model_name)
        grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
        return torch.cat(grads).norm()

    def log_round(self, outputs: dict[str, Any]) -> None:
        """Write a round's metrics: the last update's losses and gradient norms, timings and peak GPU memory."""
        with self.measure_gpu("learn/logging"):
            metrics = {"train/" + key: value for key, value in outputs.items() if key.endswith("_loss")}
            metrics |= {f"gradients/{name}": self.gradient_norm(name) for name in self.GRADIENT_NORMS}
            metrics |= {f"learning_rate/{name}": rate for name, rate in self.learning_rates.items()}
        on_gpu = self.device.type == "cuda"
        if self.timings.enabled:
            if on_gpu:
                with self.timings.measure("learn/gpu_drain"):
                    torch.cuda.synchronize(self.device)
            metrics |= {f"timing/{name}_ms": milliseconds for name, milliseconds in self.timings.summary().items()}
            last_time, last_num_steps = self.last_timings_log
            now = time.perf_counter()
            metrics["timing/env_steps_per_second"] = (self.num_steps - last_num_steps) / (now - last_time)
            self.last_timings_log = (now, self.num_steps)
            if on_gpu:  # Peaks since the last round.
                metrics["memory/gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                metrics["memory/gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(self.device) / 2**30
                torch.cuda.reset_peak_memory_stats(self.device)
        self.logger.log_metrics(metrics, step=self.num_steps)

    # ------------------------------------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            "state_dict": self.state_dict(),
            "optimizer_states": [optimizer.state_dict() for optimizer in self.optimizers],
            "hyper_parameters": self.hparams,
            "num_steps": self.num_steps,
            "num_episodes": self.num_episodes,
        }
        torch.save(checkpoint | self.replay_buffer_checkpoint(), path)

    def delete_other_checkpoints(self, keep: Path) -> None:
        """Delete the run's checkpoint files except keep (called once the new one is fully written)."""
        for checkpoint in self.checkpoints_dir.glob("*.ckpt"):
            if checkpoint != keep:
                checkpoint.unlink()

    def replay_buffer_checkpoint(self) -> dict[str, Any]:
        """Describe the replay buffer for resuming: its bookkeeping and the directory of its files, which with
        save_replay_buffer is a copy next to the checkpoints (replacing older copies)."""
        replay_buffer = self.replay_buffer
        replay_buffer.save_bookkeeping()  # Now describing every stored episode (add_episode leaves out the newest).
        replay_buffer_dir = Path(replay_buffer.save_path).resolve()
        if self.save_replay_buffer:
            copy_dir = self.checkpoints_dir / f"replay_buffer-steps={self.num_steps}-episode={self.num_episodes}"
            shutil.copytree(replay_buffer_dir, copy_dir)
            # Keep only the copy for this checkpoint (sorting the names as text would misorder 9000 and 10500).
            for old_copy_dir in self.checkpoints_dir.glob("replay_buffer-steps=*"):
                if old_copy_dir != copy_dir:
                    shutil.rmtree(old_copy_dir)
            replay_buffer_dir = copy_dir
        return {
            "replay_buffer_path": replay_buffer_dir,
            "replay_buffer_capacity": replay_buffer.capacity,
            "replay_buffer_info": replay_buffer.info,
            "replay_buffer_head": replay_buffer.head,
            "replay_buffer_tail": replay_buffer.tail,
            "replay_buffer_episode_boundaries": replay_buffer.episode_boundaries,
        }

    @classmethod
    def load(cls, path: str | Path, env: gym.Env, map_location: str | torch.device = "cpu", **overrides: Any):
        """The module saved at path, rebuilt from the checkpoint's hyperparameters (overrides replace some)."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        module = cls(env=env, **{**checkpoint["hyper_parameters"], **overrides})
        module.load_state_dict(checkpoint["state_dict"])
        module.load_counters(checkpoint)
        return module

    def load_counters(self, checkpoint: dict[str, Any]) -> None:
        self.num_steps = checkpoint["num_steps"]
        self.num_episodes = checkpoint["num_episodes"] - self.num_envs  # The first reset adds num_envs again.

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Resume training from a checkpoint: weights, optimizer states, counters and the saved replay buffer."""
        self.load_state_dict(checkpoint["state_dict"])
        states = checkpoint["optimizer_states"]
        counts = f"{len(states)} optimizer states for {len(self.optimizers)} optimizers"
        if len(states) > len(self.optimizers):
            raise ValueError(f"The checkpoint has {counts}.")
        if len(states) < len(self.optimizers):  # E.g. fine-tuning runs saved before BUGS_AND_PITFALLS.md #17 was fixed.
            fresh = len(self.optimizers) - len(states)
            warnings.warn(f"The checkpoint has {counts}: the last {fresh} start fresh.", stacklevel=2)
        for optimizer, state in zip(self.optimizers, states):
            # Keep this run's learning rate and kernel choice; loading moves the step counters to where fused kernels
            # need them.
            groups = [
                {**saved, **{key: group[key] for key in ("lr", "fused", "foreach") if key in group}}
                for saved, group in zip(state["param_groups"], optimizer.param_groups)
            ]
            optimizer.load_state_dict({**state, "param_groups": groups})
        self.load_counters(checkpoint)
        self.num_updates_since_restore = 0

        if "replay_buffer_path" in checkpoint:
            self.restore_replay_buffer(checkpoint)

    def restore_replay_buffer(self, checkpoint: dict[str, Any]) -> None:
        """Load the replay buffer files the checkpoint points to, described by their bookkeeping.pt (consistent with
        the files and, for a live buffer, possibly newer than the checkpoint) or, for older checkpoints without that
        file, by the checkpoint's replay_buffer_* keys."""
        path = Path(checkpoint["replay_buffer_path"])
        if not path.exists():
            warnings.warn(f"The replay buffer {path} no longer exists: resuming with an empty one.", stacklevel=2)
            return
        if (path / BOOKKEEPING_FILE).exists():
            bookkeeping, source = RingBufferDataset.load_bookkeeping(path), str(path / BOOKKEEPING_FILE)
        elif "replay_buffer_info" in checkpoint:
            keys = ("capacity", "info", "head", "tail", "episode_boundaries")
            bookkeeping, source = {key: checkpoint[f"replay_buffer_{key}"] for key in keys}, "the checkpoint"
        else:
            raise ValueError(
                f"{path} has no {BOOKKEEPING_FILE} and the checkpoint does not describe the replay buffer: rebuild "
                "the bookkeeping with scripts/rebuild_replay_buffer_bookkeeping.py."
            )
        if bookkeeping["capacity"] != self.replay_buffer.capacity:
            raise ValueError(
                f"The saved replay buffer holds {bookkeeping['capacity']} steps, "
                f"but this run's buffer_capacity is {self.replay_buffer.capacity}."
            )
        info = bookkeeping["info"]
        if info.is_empty:  # E.g. the checkpoint from before the first episode ended.
            return
        if info.fields is None:
            raise ValueError("Cannot load replay buffer without fields.")
        self.replay_buffer.load_from_disk_storage(path, bookkeeping)
        logger.info(
            "Restored %d episodes (%d steps) of the replay buffer from %s, described by %s.",
            info.num_episodes,
            info.num_timesteps,
            path,
            source,
        )
