"""envs/vector.py: batches of environments."""

from sold.envs.vector import SingleEnv
from tests import fakes
import torch


def test_single_env_is_a_batch_of_one():
    env = fakes.make_fake_env()
    single = SingleEnv(env)
    assert single.num_envs == 1 and single.action_space is env.action_space and single.max_episode_steps == 8
    assert single.reset().shape == (1, 3, 16, 16)
    obs, rewards, dones, infos = single.step(torch.zeros(1, 2))
    assert obs.shape == (1, 3, 16, 16) and isinstance(rewards[0], float) and dones == [False] and "success" in infos[0]
    assert single.render(size=(4, 4)).shape == (4, 4, 3)  # Other attributes are the environment's.
    single.close()
    assert env.unwrapped.closed
