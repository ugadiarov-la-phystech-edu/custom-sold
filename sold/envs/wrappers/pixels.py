import gym
import numpy as np
from PIL import Image
from typing import Any


class Pixels(gym.Wrapper):
    """Replaces the observations by rendered frames (3, height, width); with an instruction memory (see
    wrappers.language), by {"image": frame, "tokens": memory}."""

    def __init__(self, env: gym.Env, image_size: tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        image_space = gym.spaces.Box(low=0, high=1, shape=(3,) + image_size, dtype=float)
        inner_spaces = getattr(env.observation_space, "spaces", {})
        self.keeps_tokens = "tokens" in inner_spaces
        self.observation_space = image_space
        if self.keeps_tokens:
            self.observation_space = gym.spaces.Dict({"image": image_space, "tokens": inner_spaces["tokens"]})
        self.renders_at_size = _is_multi_object_fetch(env.unwrapped)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info

    def reset(self) -> Any:
        return self._observation(self.env.reset())

    def _observation(self, obs: Any) -> Any:
        if self.keeps_tokens:
            return {"image": self._get_obs(), "tokens": obs["tokens"]}
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.renders_at_size:
            image = self.env.render(mode="rgb_array", size=self.image_size)
        else:
            height, width = self.image_size
            image = Image.fromarray(self.env.render(mode="rgb_array"))
            image = np.array(image.resize((width, height)))  # PIL sizes are (width, height).
        return np.moveaxis(image, -1, 0)


def _is_multi_object_fetch(env: gym.Env) -> bool:
    """Imported lazily so that the gym and dmcontrol suites work without multi_object_fetch installed."""
    try:
        from multi_object_fetch.env import MultiObjectFetchEnv
    except ImportError:
        return False
    return isinstance(env, MultiObjectFetchEnv)
