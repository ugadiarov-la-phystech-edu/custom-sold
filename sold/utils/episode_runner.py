from collections import defaultdict
from collections.abc import Callable
import gym
import torch
from typing import Any


class EpisodeRunner:
    """Steps a policy through an environment, one step or one whole episode at a time.

    The policy is called as policy(obs, is_first=..., mode=...) and returns the action.
    """

    def __init__(self, env: gym.Env, policy: Callable[..., torch.Tensor]) -> None:
        self.env = env
        self.policy = policy
        self.obs: torch.Tensor | None = None
        self.done = True  # No episode in progress.
        self.is_first = True  # The next step is the first of its episode.

    def reset(self) -> torch.Tensor:
        """Start a new episode and return its first observation."""
        self.obs, self.done, self.is_first = self.env.reset(), False, True
        return self.obs

    def step(self, mode: str) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Take an action chosen in mode ("train", "eval" or "random"); return obs, reward, done, info."""
        action = self.policy(self.obs, is_first=self.is_first, mode=mode)
        self.obs, reward, self.done, info = self.env.step(action.cpu())
        self.is_first = False
        return self.obs, reward, self.done, info

    @torch.no_grad()
    def play_episode(
        self, mode: str = "eval", render: Callable[[gym.Env], torch.Tensor] | None = None
    ) -> dict[str, Any]:
        """Play a whole episode without storing experience.

        Returns the observations, rewards, final "success" (if reported) and, with render, a frame after every reset
        and step under "rendered".
        """
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before playing an episode.")

        episode, info = defaultdict(list), {}

        def record(obs: torch.Tensor) -> None:
            episode["obs"].append(obs.cpu())
            if render is not None:
                episode["rendered"].append(render(self.env))

        record(self.reset())
        while not self.done:
            obs, reward, _, info = self.step(mode)
            episode["reward"].append(reward)
            record(obs)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
