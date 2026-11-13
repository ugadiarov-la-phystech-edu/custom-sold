from collections.abc import Mapping
import gym
import os
from sold.envs.from_gym import make_env as make_gym_env
from sold.envs.wrappers.to_tensor import ToTensor
from typing import Optional


def make_env(
    suite: str,
    name: str,
    image_size: tuple[int, int],
    max_episode_steps: int,
    action_repeat: int,
    seed: int | None = None,
    language: Mapping | None = None,
) -> gym.Env:
    """language (multi_object_fetch tasks, suites gym and mof): stream the task's instruction into a memory (see
    wrappers.language); the observations are then {"image", "tokens"}. None or empty: no instruction."""
    os.environ["MUJOCO_GL"] = "egl"  # MuJoCo rendering backend; set before importing a MuJoCo suite.
    if language and suite not in ("gym", "mof"):
        raise ValueError(f"Streamed instructions need the 'gym' or 'mof' suite, not {suite!r}.")
    if suite == "gym":
        env = make_gym_env(name, image_size, max_episode_steps, action_repeat, seed, language=language)
    elif suite == "mof":
        from sold.envs.from_mof import make_env as make_mof_env

        env = make_mof_env(name, image_size, max_episode_steps, action_repeat, seed, language=language)
    elif suite == "dmcontrol":
        from sold.envs.from_dmcontrol import make_env as make_dmcontrol_env

        env = make_dmcontrol_env(name, image_size, max_episode_steps, action_repeat, seed)
    elif suite == "survival":
        try:
            from sold.envs.from_survival import make_env as make_survival_env
        except ModuleNotFoundError as error:
            raise ValueError(
                "The 'survival' suite needs envs/from_survival.py, which is not part of this repository."
            ) from error
        from sold.envs.wrappers.to_tensordict import ToTensorDict

        env = make_survival_env(name, image_size, max_episode_steps, action_repeat, seed)
        env = ToTensorDict(env)
    else:
        raise ValueError(f"Unsupported environment suite: {suite}")
    env = ToTensor(env)
    return env
