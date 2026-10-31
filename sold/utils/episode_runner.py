from collections import defaultdict
from collections.abc import Callable
import gym
from sold.utils.profiling import Timings
import torch
from typing import Any


class EpisodeRunner:
    """Steps a policy through an environment, one step or one whole episode at a time.

    The policy is called as policy(obs, is_first=..., mode=...) and returns the action. timings records env_reset,
    action_selection (with the action's copy to the CPU) and env_step under the phase "collect" or "eval".
    """

    def __init__(self, env: gym.Env, policy: Callable[..., torch.Tensor], timings: Timings | None = None) -> None:
        self.env = env
        self.policy = policy
        self.timings = timings or Timings(enabled=False)
        self.obs: torch.Tensor | None = None
        self.action: torch.Tensor | None = None  # The last action taken, on the CPU.
        self.done = True  # No episode in progress.
        self.is_first = True  # The next step is the first of its episode.

    def reset(self, phase: str = "collect") -> torch.Tensor:
        """Start a new episode and return its first observation."""
        with self.timings.measure(f"{phase}/env_reset"):
            self.obs = self.env.reset()
        self.done, self.is_first = False, True
        return self.obs

    def step(self, mode: str, phase: str = "collect") -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Take an action chosen in mode ("train", "eval" or "random"); return obs, reward, done, info."""
        with self.timings.measure(f"{phase}/action_selection"):
            self.action = self.policy(self.obs, is_first=self.is_first, mode=mode).cpu()
        with self.timings.measure(f"{phase}/env_step"):
            self.obs, reward, self.done, info = self.env.step(self.action)
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

        episode: defaultdict[str, list[Any]] = defaultdict(list)
        info: dict[str, Any] = {}

        def record(obs: torch.Tensor) -> None:
            episode["obs"].append(obs.cpu())
            if render is not None:
                episode["rendered"].append(render(self.env))

        record(self.reset(phase="eval"))
        while not self.done:
            obs, reward, _, info = self.step(mode, phase="eval")
            episode["reward"].append(reward)
            record(obs)

        if "success" in info:
            episode["success"] = info["success"]
        return episode
