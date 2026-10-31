"""envs: wrappers, make_env for every suite, multi_object_fetch distractor sampling, dm_control adapters."""

import importlib
import sys

import numpy as np
import pytest
import torch

import envs
from envs.wrappers.action_repeat import ActionRepeat
from envs.wrappers.pixels import Pixels
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.to_tensor import ToTensor
from tests import fakes


def test_importing_envs_selects_egl():
    import os
    assert os.environ["MUJOCO_GL"] == "egl"


def test_action_repeat_accumulates_reward_and_stops_on_done():
    env = ActionRepeat(fakes.FakePixelEnv(episode_length=3), action_repeat=2)
    env.reset()
    _, reward, done, _ = env.step(np.ones(2))
    assert env.unwrapped.t == 2 and not done
    assert reward == pytest.approx((1 - abs(0.1 - 0.5) * 2) + (1 - abs(0.2 - 0.5) * 2))
    _, _, done, _ = env.step(np.ones(2))
    assert env.unwrapped.t == 3 and done  # Stopped after one repeat.


def test_time_limit_truncates_and_resets():
    env = TimeLimit(fakes.FakePixelEnv(episode_length=10), max_episode_steps=2)
    env.reset()
    assert env.step(np.zeros(2))[2] is False
    _, _, done, info = env.step(np.zeros(2))
    assert done and info["TimeLimit.truncated"] is True
    env.reset()
    assert env._elapsed_steps == 0
    finished = TimeLimit(fakes.FakePixelEnv(episode_length=1), max_episode_steps=1)
    finished.reset()
    assert finished.step(np.zeros(2))[3]["TimeLimit.truncated"] is False


def test_pixels_resizes_rendered_frames():
    env = Pixels(fakes.FakePixelEnv(), image_size=(8, 6))
    obs = env.reset()
    assert obs.shape == (3, 8, 6) and obs.dtype == np.uint8  # (3, height, width), also when not square (#24).
    obs, reward, done, info = env.step(np.ones(2))
    assert obs.shape == (3, 8, 6) and "success" in info
    assert env.observation_space.shape == (3, 8, 6)
    square = Pixels(fakes.FakePixelEnv(), image_size=(8, 8))
    assert square.reset().shape == square.observation_space.shape


def test_pixels_renders_multi_object_fetch_at_target_size():
    from multi_object_fetch.env import MultiObjectFetchEnv

    class FakeMof(fakes.FakePixelEnv):
        render_sizes = []

        @property
        def unwrapped(self):
            concrete = type("ConcreteMof", (MultiObjectFetchEnv,), {"success": lambda self: False})
            return concrete.__new__(concrete)

        def render(self, mode="rgb_array", size=None):
            self.render_sizes.append(size)
            return np.zeros((*size, 3), np.uint8)

    env = Pixels(FakeMof(), image_size=(5, 7))
    assert env.reset().shape == (3, 5, 7)
    assert FakeMof.render_sizes == [(5, 7)]


def test_to_tensor_converts_observations_and_actions():
    env = ToTensor(Pixels(fakes.FakePixelEnv(), (4, 4)))
    obs = env.reset()
    assert isinstance(obs, torch.Tensor) and obs.shape == (3, 4, 4)
    obs, _, _, _ = env.step(torch.ones(2, requires_grad=True))
    assert isinstance(obs, torch.Tensor)
    assert isinstance(env.unwrapped.actions[-1], np.ndarray)


def test_make_env_gym_suite_builds_the_wrapper_stack():
    env = envs.make_env("gym", "FakePixel-v0", (16, 16), max_episode_steps=5, action_repeat=2)
    assert isinstance(env, ToTensor) and isinstance(env.env, Pixels)
    assert isinstance(env.env.env, TimeLimit) and isinstance(env.env.env.env, ActionRepeat)
    assert env.max_episode_steps == 5
    obs, done, steps = env.reset(), False, 0
    while not done:
        obs, _, done, _ = env.step(torch.zeros(2))
        steps += 1
    assert steps == 5 and env.unwrapped.t == 10 and obs.dtype == torch.uint8


def test_make_env_rejects_unknown_suite():
    with pytest.raises(ValueError, match="Unsupported environment suite"):
        envs.make_env("atari", "x", (8, 8), 5, 1)


def test_make_env_mof_suite(monkeypatch):
    import envs.from_mof as from_mof
    created = []

    def fake_make_gym_env(name, image_size, max_episode_steps, action_repeat, seed):
        created.append((name, image_size, max_episode_steps, action_repeat, seed))
        return Pixels(TimeLimit(fakes.FakePixelEnv(), max_episode_steps), image_size)

    monkeypatch.setattr(from_mof, "make_gym_env", fake_make_gym_env)
    env = envs.make_env("mof", "PushRed_1to3Distractors_Dense-v1", (8, 8), 50, 2, seed=5)
    assert isinstance(env, ToTensor) and isinstance(env.env, from_mof.VariableDistractorsWrapper)
    assert created[0][1:4] == ((8, 8), 50, 2)
    for _ in range(30):
        assert env.reset().shape == (3, 8, 8)
    names = [name for name, *_ in created]
    # One environment per number of distractors, created on first use and then reused (#12).
    assert sorted(names) == [f"PushRed_{n}Distractors_Dense-v1" for n in (1, 2, 3)]
    assert sorted(seed for *_, seed in created) == [6, 7, 8]  # seed + number of distractors (#11).
    wrapper = env.env
    env.close()
    assert all(inner.unwrapped.closed for inner in wrapper.envs.values())


def test_make_env_mof_suite_without_seed(monkeypatch):
    import envs.from_mof as from_mof
    seeds = []
    monkeypatch.setattr(from_mof, "make_gym_env", lambda name, size, steps, repeat, seed: (
        seeds.append(seed), Pixels(fakes.FakePixelEnv(), size))[1])
    envs.make_env("mof", "PushRed_2to2Distractors_Dense-v1", (8, 8), 50, 2)
    assert seeds == [None]


def test_make_env_gym_suite_seeds_the_environment_only_when_asked():
    assert envs.make_env("gym", "FakePixel-v0", (8, 8), 5, 1, seed=11).unwrapped.seed_value == 11
    assert envs.make_env("gym", "FakePixel-v0", (8, 8), 5, 1).unwrapped.seed_value is None


def test_make_env_survival_suite_with_fake_module(monkeypatch):
    survival = fakes.install_fake_survival_suite(monkeypatch)
    env = envs.make_env("survival", "Forage", (8, 8), 10, 1, seed=3)
    assert survival.calls == [("Forage", (8, 8), 10, 1, 3)]
    from envs.wrappers.to_tensordict import ToTensorDict
    assert isinstance(env, ToTensor) and isinstance(env.env, ToTensorDict)


def test_to_tensor_dict_with_fake_modules(monkeypatch):
    fakes.install_fake_gymnasium_and_tensordict(monkeypatch)
    to_tensordict = importlib.import_module("envs.wrappers.to_tensordict")
    inner = fakes.FakeGymnasiumEnv()
    env = to_tensordict.ToTensorDict(inner)
    obs, info = env.reset()
    assert info == {"reset": True}
    assert obs["image"].shape == (3, 8, 8) and obs["image"].max() == 1.0
    assert torch.equal(obs["state"], torch.tensor([0.0, 1.0, 2.0]))
    obs, reward, terminated, truncated, info = env.step(torch.ones(2, requires_grad=True))
    assert (reward, terminated, truncated, info) == (0.5, False, True, {"step": True})
    assert isinstance(obs, fakes.FakeTensorDict) and isinstance(inner.received_actions[0], np.ndarray)


# BUGS_AND_PITFALLS.md #8 (fixed).
def test_survival_suite_is_usable_or_rejected_cleanly():
    try:
        envs.make_env("survival", "Forage", (16, 16), 10, 1)
    except ValueError:
        pass  # Removing the suite is a valid fix.


# BUGS_AND_PITFALLS.md #12 (fixed).
def test_distractor_wrapper_does_not_leak_environments(monkeypatch):
    import envs.from_mof as from_mof
    created = []

    def fake_make_gym_env(name, image_size, max_episode_steps, action_repeat, seed):
        created.append(fakes.FakePixelEnv())
        return Pixels(created[-1], image_size)

    monkeypatch.setattr(from_mof, "make_gym_env", fake_make_gym_env)
    env = from_mof.make_env("PushRed_1to3Distractors_Dense-v1", (8, 8), 50, 2)
    for _ in range(30):
        env.reset()
    alive = [inner for inner in created if not inner.closed]
    assert len(alive) <= 3, f"{len(alive)} environments are still open after 30 resets"


# BUGS_AND_PITFALLS.md #15 (fixed).
def test_pixels_wrapper_does_not_require_multi_object_fetch(monkeypatch):
    monkeypatch.setitem(sys.modules, "multi_object_fetch", None)
    monkeypatch.setitem(sys.modules, "multi_object_fetch.env", None)
    monkeypatch.delitem(sys.modules, "envs.wrappers.pixels")
    pixels = importlib.import_module("envs.wrappers.pixels")
    assert pixels.Pixels(fakes.FakePixelEnv(), (8, 8)).reset().shape == (3, 8, 8)


# BUGS_AND_PITFALLS.md #24 (fixed).
def test_pixels_observation_matches_observation_space_for_non_square_sizes():
    env = Pixels(fakes.FakePixelEnv(), image_size=(8, 6))
    assert env.reset().shape == env.observation_space.shape



# ------------------------------------------------------------------------------- dm_control
@pytest.fixture
def dmc(monkeypatch):
    loaded = fakes.install_fake_dm_control(monkeypatch)
    module = importlib.import_module("envs.from_dmcontrol")
    return module, loaded


def test_dmcontrol_make_env_validates_task(dmc):
    module, _ = dmc
    with pytest.raises(ValueError, match="Unknown dm-control task"):
        module.make_env("cheetah-run", (8, 8), 10, 2)


def test_dmcontrol_make_env_full_episode(dmc):
    module, loaded = dmc
    env = module.make_env("cup-catch", (8, 8), max_episode_steps=10, action_repeat=2, seed=7)
    assert (loaded[0].domain, loaded[0].task, loaded[0].seed) == ("ball_in_cup", "catch", 7)
    assert isinstance(env, Pixels) and env.max_episode_steps == 10
    gym_env = env.env.env
    assert isinstance(gym_env, module.TimeStepToGymWrapper)
    assert gym_env.observation_space.shape == (3,)
    assert gym_env.action_space.low.tolist() == [-1.0, -1.0] and gym_env.action_space.dtype == np.float32
    assert gym_env.reward_range is None and gym_env.metadata is None
    assert isinstance(gym_env.unwrapped, module.ExtendedTimeStepWrapper)

    obs = env.reset()
    assert obs.shape == (3, 8, 8) and loaded[0].physics.render_calls[-1] == (384, 384, 0)
    _, reward, done, info = env.step(np.ones(2, np.float32))
    assert reward == pytest.approx(2.0) and not done and info["anything"] == 0.0
    assert loaded[0].received_actions[0].dtype == np.float64
    assert np.allclose(loaded[0].received_actions[0], [2.0, 2.0])  # Rescaled from [-1, 1] to [-2, 2].
    _, reward, done, _ = env.step(np.ones(2, np.float32))
    assert done and reward == pytest.approx(1.0)  # LAST after 3 inner steps; repeat stopped early.


def test_dmcontrol_quadruped_uses_camera_two(dmc):
    module, loaded = dmc
    env = module.make_env("quadruped-walk", (8, 8), 10, 1)
    env.reset()
    assert loaded[0].physics.render_calls[-1][2] == 2


def test_dmcontrol_gym_wrapper_episode_limit(dmc):
    module, _ = dmc
    env = module.make_env("cartpole-swingup", (8, 8), 100, 1)
    gym_env = env.env.env
    gym_env.max_episode_steps = 1
    gym_env.reset()
    assert gym_env.step(np.zeros(2, np.float32))[2] is True


def test_dmcontrol_wrappers_forward_specs_and_attributes(dmc):
    module, _ = dmc
    inner = fakes.FakeDmEnv()
    dtype_env = module.ActionDTypeWrapper(inner, np.float32)
    assert dtype_env.action_spec().dtype == np.float32 and "position" in dtype_env.observation_spec()
    assert dtype_env.reset().first() and dtype_env.physics is inner.physics
    repeat_env = module.ActionRepeatWrapper(dtype_env, 3)
    assert repeat_env.action_spec().dtype == np.float32 and "position" in repeat_env.observation_spec()
    assert repeat_env.reset().first() and repeat_env.physics is inner.physics
    step = repeat_env.step(np.zeros(2, np.float32))
    assert step.last() and step.reward == pytest.approx(3.0) and step.discount == pytest.approx(0.5)

    extended = module.ExtendedTimeStepWrapper(fakes.FakeDmEnv())
    first = extended.reset()
    assert first.first() and not first.mid() and not first.last()
    assert first.reward == 0.0 and first.discount == 1.0 and first.action.tolist() == [0.0, 0.0]
    mid = extended.step(np.ones(2))
    assert mid.mid() and mid.action.tolist() == [1.0, 1.0]
    assert "position" in extended.observation_spec() and extended.physics is not None


def test_make_env_dmcontrol_suite(dmc):
    env = envs.make_env("dmcontrol", "cartpole-swingup", (8, 8), 10, 2, seed=1)
    assert isinstance(env, ToTensor) and env.reset().shape == (3, 8, 8)
