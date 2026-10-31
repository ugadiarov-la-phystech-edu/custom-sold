"""The real multi_object_fetch environments through SOLD's wrapper stack (needs mujoco-py with rendering)."""

import pytest
import torch

import envs
from envs.from_mof import VariableDistractorsWrapper

pytestmark = pytest.mark.mujoco


@pytest.mark.parametrize("name", ["ReachRed_0to4Distractors_Dense-v1", "PushRed_0to4Distractors_Dense-v1",
                                  "PickRed_0to4Distractors_Dense-v1"])
def test_mof_environment_through_sold_wrappers(name):
    env = envs.make_env("mof", name, (64, 64), max_episode_steps=50, action_repeat=2)
    assert isinstance(env.env, VariableDistractorsWrapper)
    obs = env.reset()
    assert obs.shape == (3, 64, 64) and obs.dtype == torch.uint8 and obs.float().std() > 0
    obs, reward, done, info = env.step(torch.zeros(env.action_space.shape))
    assert obs.shape == (3, 64, 64) and isinstance(reward, float) and "success" in info


def test_episode_lengths_per_task():
    """Characterizes pitfall #9 (gym's own time limit): Reach ends after 25 agent steps, Push after 50."""
    lengths = {}
    for task in ["ReachRed", "PushRed"]:
        env = envs.make_env("mof", f"{task}_0to0Distractors_Dense-v1", (32, 32), 50, 2)
        env.reset()
        done, steps = False, 0
        while not done:
            _, _, done, _ = env.step(torch.zeros(env.action_space.shape))
            steps += 1
        lengths[task] = steps
    assert lengths == {"ReachRed": 25, "PushRed": 50}


@pytest.mark.xfail(strict=True, reason="BUGS_AND_PITFALLS.md #9: Reach episodes end after 25 steps (gym's own time limit)")
def test_reach_episodes_last_max_episode_steps():
    env = envs.make_env("mof", "ReachRed_0to0Distractors_Dense-v1", (64, 64), max_episode_steps=50, action_repeat=2)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(torch.zeros(env.action_space.shape))
        steps += 1
    assert steps == 50
