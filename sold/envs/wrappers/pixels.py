import gym
from PIL import Image
import numpy as np
from typing import Tuple


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,) + image_size, dtype=float)
        self.renders_at_size = _is_multi_object_fetch(env.unwrapped)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
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
