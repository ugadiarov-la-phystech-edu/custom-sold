import gym
from sold.envs.wrappers.action_repeat import ActionRepeat
from sold.envs.wrappers.pixels import Pixels
from sold.envs.wrappers.time_limit import TimeLimit


def make_env(
    name: str, image_size: tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int | None = None
):
    env = gym.make(name)
    if seed is not None:  # Without a seed, gym environments seed themselves randomly.
        env.seed(seed)
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    return env
