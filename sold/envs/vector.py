"""Batches of environments: reset and step take and return one entry per environment (observations as tensors, or dicts
of them for dict observations such as {"image", "tokens"}).

SingleEnv runs one environment in-process as a batch of one; VectorEnv runs each environment in its own process, so that
their steps run in parallel. Both expose num_envs, the single environment's action_space and max_episode_steps.
"""

from collections.abc import Callable, Sequence
import gym
import multiprocessing
from multiprocessing.connection import Connection
import numpy as np
import random
import signal
import torch
import traceback
from typing import Any


class SingleEnv:
    """One environment as a batch of one; every other attribute (action_space, render, ...) is the environment's."""

    num_envs = 1

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self) -> Any:
        """The first observations, shape (1, ...)."""
        return stack_observations([self.env.reset()])

    def step(self, actions: torch.Tensor) -> tuple[Any, list[float], list[bool], list[dict[str, Any]]]:
        """actions: (1, action_dim). Returns the observations (1, ...) and lists of the rewards, dones and infos."""
        obs, reward, done, info = self.env.step(actions[0])
        return stack_observations([obs]), [reward], [done], [info]


def stack_observations(observations: Sequence[Any]) -> Any:
    """A batch of the environments' observations (tensors or numpy arrays): a tensor, or a dict of them for dict
    observations."""
    if isinstance(observations[0], dict):
        return {key: stack_observations([obs[key] for obs in observations]) for key in observations[0]}
    if isinstance(observations[0], torch.Tensor):
        return torch.stack(list(observations))
    return torch.from_numpy(np.stack(observations))


def to_numpy(obs: Any) -> Any:
    """An observation (a tensor or a dict of them) as numpy arrays, to send between processes."""
    return {key: value.numpy() for key, value in obs.items()} if isinstance(obs, dict) else obs.numpy()


def env_worker(connection: Connection, env_factory: Callable[..., gym.Env], seed: int) -> None:
    """Build env_factory(seed=seed) and serve "reset", "step" and "close" requests until "close" or an error.

    Replies are ("ok", result) or ("error", traceback). Arrays travel as numpy arrays. The worker seeds Python's and
    numpy's generators with seed too (multi_object_fetch picks the number of distractors with random), and ignores
    Ctrl-C, which the trainer handles.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        random.seed(seed)
        np.random.seed(seed)
        env = env_factory(seed=seed)
        connection.send(("ok", (env.action_space, env.max_episode_steps)))
        while True:
            command, data = connection.recv()
            if command == "reset":
                connection.send(("ok", to_numpy(env.reset())))
            elif command == "step":
                obs, reward, done, info = env.step(torch.from_numpy(data))
                connection.send(("ok", (to_numpy(obs), reward, done, info)))
            elif command == "close":
                env.close()
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"Unknown command: {command}")
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class VectorEnv:
    """num_envs environments, each built by env_factory(seed=seed + 1000 * index) in its own process.

    The processes are started with "spawn": fresh interpreters, each with its own rendering context and without the
    trainer's CUDA state. Steps are sent to all environments before the results are gathered, so they run in parallel.
    A worker's error is raised in the trainer with the worker's traceback.
    """

    def __init__(self, env_factory: Callable[..., gym.Env], num_envs: int, seed: int = 0) -> None:
        context = multiprocessing.get_context("spawn")
        self.num_envs = num_envs
        self.connections: list[Connection] = []
        self.processes: list[Any] = []
        self.closed = False
        for index in range(num_envs):
            connection, child = context.Pipe()
            process = context.Process(
                target=env_worker, args=(child, env_factory, seed + 1000 * index), daemon=True, name=f"env-{index}"
            )
            process.start()
            child.close()
            self.connections.append(connection)
            self.processes.append(process)
        self.action_space, self.max_episode_steps = self._gather()[0]
        self.action_space.seed(seed)  # The trainer samples random actions from it.

    def _gather(self) -> list[Any]:
        results = []
        for index, connection in enumerate(self.connections):
            status, result = connection.recv()
            if status == "error":
                self.close()
                raise RuntimeError(f"Environment worker {index} failed:\n{result}")
            results.append(result)
        return results

    def reset(self) -> Any:
        """The first observations, (num_envs, ...)."""
        for connection in self.connections:
            connection.send(("reset", None))
        return stack_observations(self._gather())

    def step(self, actions: torch.Tensor) -> tuple[Any, list[float], list[bool], list[dict[str, Any]]]:
        """actions: (num_envs, action_dim). Returns the observations (num_envs, ...) and lists of the rewards, dones and
        infos."""
        for connection, action in zip(self.connections, actions.detach().cpu().numpy()):
            connection.send(("step", action))
        obs, rewards, dones, infos = zip(*self._gather())
        return stack_observations(obs), list(rewards), list(dones), list(infos)

    def close(self) -> None:
        """Close the environments and end the processes (terminating those that do not end within 10 s)."""
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):  # The worker has already ended.
                pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for connection in self.connections:
            connection.close()
