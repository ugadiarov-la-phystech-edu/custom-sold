import gym
import numpy as np
import torch
from typing import Any


class ToTensor(gym.Wrapper):
    """Observations as tensors (dicts of them for dict observations); actions to numpy arrays."""

    def step(self, action: torch.Tensor) -> tuple[Any, float, bool, dict]:
        obs, reward, done, info = self.env.step(action.detach().cpu().numpy())
        return to_tensor(obs), reward, done, info

    def reset(self) -> Any:
        return to_tensor(self.env.reset())


def to_tensor(obs: np.ndarray | dict[str, np.ndarray]) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(obs, dict):
        return {key: torch.from_numpy(value) for key, value in obs.items()}
    return torch.from_numpy(obs)
