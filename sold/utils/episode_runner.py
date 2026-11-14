from collections import defaultdict
from collections.abc import Callable
from sold.utils.profiling import Timings
import torch
from typing import Any


class EpisodeRunner:
    """Steps a policy through a batch of environments (sold.envs.vector), one step or whole episodes at a time.

    The policy is called as policy(obs, is_first=..., mode=...) with the observations of all environments and returns
    their actions. The environments' episodes run in lockstep: they all end at the same step. timings records
    env_reset, action_selection (with the actions' copy to the CPU) and env_step under the phase "collect" or "eval".
    Observations are images, or {"image", "tokens"} with an instruction memory; played episodes record the images.
    """

    def __init__(self, env: Any, policy: Callable[..., torch.Tensor], timings: Timings | None = None) -> None:
        self.env = env
        self.policy = policy
        self.timings = timings or Timings(enabled=False)
        self.obs: Any = None
        self.action: torch.Tensor | None = None  # The last actions taken, (num_envs, action_dim) on the CPU.
        self.done = True  # No episode in progress.
        self.is_first = True  # The next step is the first of its episode.

    def reset(self, phase: str = "collect") -> Any:
        """Start new episodes and return their first observations, (num_envs, ...)."""
        with self.timings.measure(f"{phase}/env_reset"):
            self.obs = self.env.reset()
        self.done, self.is_first = False, True
        return self.obs

    def step(self, mode: str, phase: str = "collect") -> tuple[Any, list[float], list[bool], list[dict[str, Any]]]:
        """Take actions chosen in mode ("train", "eval" or "random"); return the obs, rewards, dones and infos."""
        with self.timings.measure(f"{phase}/action_selection"):
            self.action = self.policy(self.obs, is_first=self.is_first, mode=mode).cpu()
        with self.timings.measure(f"{phase}/env_step"):
            self.obs, rewards, dones, infos = self.env.step(self.action)
        if len(set(dones)) > 1:
            raise RuntimeError(f"The environments' episodes must end together, got dones {dones}.")
        self.done, self.is_first = dones[0], False
        return self.obs, rewards, dones, infos

    @torch.no_grad()
    def play_episodes(
        self, mode: str, num_episodes: int, render: Callable[[Any], torch.Tensor] | None = None
    ) -> list[dict[str, Any]]:
        """Play num_episodes whole episodes, in batches of num_envs, without storing experience.

        Each episode has the observations, rewards, final "success" (if reported) and, with render (one environment
        only), a frame after every reset and step under "rendered".
        """
        if not self.done:
            raise RuntimeError("Current training episode must have terminated before playing an episode.")
        if render is not None and self.env.num_envs != 1:
            raise ValueError("Rendering needs a single environment.")

        episodes: list[dict[str, Any]] = []
        while len(episodes) < num_episodes:
            batch: list[defaultdict[str, list[Any]]] = [defaultdict(list) for _ in range(self.env.num_envs)]
            self._record(batch, self.reset(phase="eval"), render)
            infos: list[dict[str, Any]] = [{} for _ in batch]
            while not self.done:
                obs, rewards, _, infos = self.step(mode, phase="eval")
                for episode, reward in zip(batch, rewards):
                    episode["reward"].append(reward)
                self._record(batch, obs, render)
            for episode, info in zip(batch, infos):
                if "success" in info:
                    episode["success"] = info["success"]
            episodes.extend(batch)
        return episodes[:num_episodes]

    def _record(
        self, batch: list[defaultdict[str, list[Any]]], obs: Any, render: Callable[[Any], torch.Tensor] | None
    ) -> None:
        images = obs["image"] if isinstance(obs, dict) else obs
        for episode, env_obs in zip(batch, images.cpu()):
            episode["obs"].append(env_obs)
        if render is not None:
            batch[0]["rendered"].append(render(self.env))

    def play_episode(self, mode: str = "eval", render: Callable[[Any], torch.Tensor] | None = None) -> dict[str, Any]:
        """One whole episode (of the first environment), as play_episodes."""
        return self.play_episodes(mode, 1, render)[0]
