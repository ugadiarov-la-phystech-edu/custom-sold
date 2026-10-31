from collections.abc import Callable
import gym
import numpy as np
import random
from sold.envs.from_gym import make_env as make_gym_env


class VariableDistractorsWrapper(gym.Wrapper):
    """Samples one of several environments (one per number of distractors) at every reset.

    Each environment is created on first use and reused afterwards, instead of building (and leaking) a new MuJoCo
    simulation on every reset.
    """

    def __init__(self, env_names: list[str], make_env: Callable[[str], gym.Env]) -> None:
        self.env_names = env_names
        self.make_env = make_env
        self.envs: dict[str, gym.Env] = {}
        super().__init__(self._get_env(random.choice(self.env_names)))

    def _get_env(self, name: str) -> gym.Env:
        if name not in self.envs:
            self.envs[name] = self.make_env(name)
        return self.envs[name]

    def reset(self) -> np.ndarray:
        self.env = self._get_env(random.choice(self.env_names))
        return self.env.reset()

    def close(self) -> None:
        for env in self.envs.values():
            env.close()


def make_env(
    name: str, image_size: tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int | None = None
):
    # Register MOF environments and parse the number of distractors.
    import multi_object_fetch  # noqa: F401  (registers the environments with gym)

    task, distractors, reward = name.split("_")
    min_distractors, max_distractors = map(int, distractors[: -len("Distractors")].split("to"))
    env_names = {
        f"{task}_{num_distractors}Distractors_{reward}": num_distractors
        for num_distractors in range(min_distractors, max_distractors + 1)
    }

    def make_variant(variant: str) -> gym.Env:
        variant_seed = None if seed is None else seed + env_names[variant]  # Different seed per variant.
        return make_gym_env(variant, image_size, max_episode_steps, action_repeat, variant_seed)

    return VariableDistractorsWrapper(list(env_names), make_variant)
