"""The real multi_object_fetch environments through SOLD's wrapper stack (needs mujoco-py with rendering)."""

import functools
import numpy as np
import pytest
import random
from sold import envs
from sold.envs.from_mof import VariableDistractorsWrapper
from sold.envs.vector import VectorEnv
import torch

pytestmark = pytest.mark.mujoco


@pytest.mark.parametrize(
    "name",
    ["ReachRed_0to4Distractors_Dense-v1", "PushRed_0to4Distractors_Dense-v1", "PickRed_0to4Distractors_Dense-v1"],
)
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


@pytest.mark.xfail(
    strict=True, reason="BUGS_AND_PITFALLS.md #9: Reach episodes end after 25 steps (gym's own time limit)"
)
def test_reach_episodes_last_max_episode_steps():
    env = envs.make_env("mof", "ReachRed_0to0Distractors_Dense-v1", (64, 64), max_episode_steps=50, action_repeat=2)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(torch.zeros(env.action_space.shape))
        steps += 1
    assert steps == 50


# BUGS_AND_PITFALLS.md #11 (fixed).
def test_the_seed_makes_environments_reproducible():
    first = [
        envs.make_env("mof", "PushRed_2to2Distractors_Dense-v1", (64, 64), 50, 2, seed=3).reset() for _ in range(2)
    ]
    assert torch.equal(first[0], first[1])


# BUGS_AND_PITFALLS.md #40 (fixed in multi_object_fetch).
def test_reset_frames_show_the_new_scene_when_environments_alternate():
    """With one cached environment per number of distractors, each reset's frame must show that reset's scene."""
    random.seed(0)  # Picks the number of distractors at each reset.
    env = envs.make_env("mof", "PushRed_0to4Distractors_Dense-v1", (64, 64), 50, 2, seed=42)
    for _ in range(10):
        frame = env.reset()
        rerendered = env.env.env.unwrapped.render(mode="rgb_array", size=(64, 64))
        assert torch.equal(frame, torch.from_numpy(np.ascontiguousarray(np.moveaxis(rerendered, -1, 0))))
    assert len(env.env.envs) > 1  # The resets alternated between environments.


def test_vector_env_of_real_environments():
    factory = functools.partial(envs.make_env, "mof", "PushRed_0to4Distractors_Dense-v1", (64, 64), 50, 2)
    env = VectorEnv(factory, num_envs=2, seed=5)
    try:
        first = env.reset()
        assert first.shape == (2, 3, 64, 64) and not torch.equal(first[0], first[1])  # Each worker its own scenes.
        obs, rewards, dones, infos = env.step(torch.zeros(2, *env.action_space.shape))
        assert obs.shape == (2, 3, 64, 64) and dones == [False, False] and all("success" in info for info in infos)
    finally:
        env.close()
    again = VectorEnv(factory, num_envs=2, seed=5)
    try:
        assert torch.equal(again.reset(), first)  # The seed makes the workers' scenes reproducible.
    finally:
        again.close()
