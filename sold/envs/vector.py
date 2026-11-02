"""Batches of environments: reset and step take and return one entry per environment.

SingleEnv runs one environment in-process as a batch of one.
"""

import gym
import torch
from typing import Any


class SingleEnv:
    """One environment as a batch of one; every other attribute (action_space, render, ...) is the environment's."""

    num_envs = 1

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self) -> torch.Tensor:
        """The first observations, shape (1, ...)."""
        return self.env.reset().unsqueeze(0)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, list[float], list[bool], list[dict[str, Any]]]:
        """actions: (1, action_dim). Returns the observations (1, ...) and lists of the rewards, dones and infos."""
        obs, reward, done, info = self.env.step(actions[0])
        return obs.unsqueeze(0), [reward], [done], [info]
