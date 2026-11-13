from collections.abc import Mapping
import gym
from sold.envs.wrappers.action_repeat import ActionRepeat
from sold.envs.wrappers.language import add_language
from sold.envs.wrappers.pixels import Pixels
from sold.envs.wrappers.time_limit import TimeLimit


def make_env(
    name: str,
    image_size: tuple[int, int],
    max_episode_steps: int,
    action_repeat: int,
    seed: int | None = None,
    language: Mapping | None = None,
):
    """language: the arguments of wrappers.language.add_language, for multi_object_fetch tasks that stream an
    instruction (one token per agent step, so outside the action repeat); None or empty: no instruction."""
    env = gym.make(name)
    if seed is not None:  # Without a seed, gym environments seed themselves randomly.
        env.seed(seed)
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    if language:
        env = add_language(env, **language)
    env = Pixels(env, image_size)
    return env
