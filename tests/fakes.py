"""Test doubles: a tiny pixel environment and fake versions of optional third-party modules."""

from collections import namedtuple
import enum
import importlib.machinery
import sys
import types

import gym
import numpy as np


class FakePixelEnv(gym.Env):
    """Deterministic 2-D point mass rendered as a small RGB image (old gym API, like multi_object_fetch)."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, episode_length: int = 12, low=(-1.0, -1.0), high=(1.0, 1.0), render_size=(32, 32)) -> None:
        self.episode_length = episode_length
        self.action_space = gym.spaces.Box(np.array(low, np.float32), np.array(high, np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.render_size = render_size
        self.position = np.zeros(2, np.float32)
        self.t = 0
        self.closed = False
        self.seed_value = None
        self.actions = []

    def seed(self, seed=None):
        self.seed_value = seed
        return [seed]

    def reset(self):
        self.position[:] = 0.0
        self.t = 0
        return self.position.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.position = np.clip(self.position + 0.1 * action, -1.0, 1.0).astype(np.float32)
        self.t += 1
        reward = float(1.0 - np.abs(self.position - 0.5).sum())
        done = self.t >= self.episode_length
        return self.position.copy(), reward, done, {"success": bool(self.position[0] > 0.05)}

    def render(self, mode="rgb_array", size=None):
        height, width = size if size is not None else self.render_size
        image = np.zeros((height, width, 3), np.uint8)
        row = int((self.position[1] + 1) / 2 * (height - 1))
        col = int((self.position[0] + 1) / 2 * (width - 1))
        image[..., 2] = (self.t * 20) % 256
        image[max(row - 1, 0) : row + 2, max(col - 1, 0) : col + 2, 0] = 255
        return image

    def close(self):
        self.closed = True


if "FakePixel-v0" not in gym.envs.registry.env_specs:
    gym.register(id="FakePixel-v0", entry_point="tests.fakes:FakePixelEnv")


def make_fake_env(image_size=(16, 16), max_episode_steps: int = 8, action_repeat: int = 1, seed=None, **env_kwargs):
    """The real SOLD wrapper stack (suite 'gym') around FakePixelEnv."""
    from envs import make_env

    env = make_env("gym", "FakePixel-v0", tuple(image_size), max_episode_steps, action_repeat, seed)
    for key, value in env_kwargs.items():
        setattr(env.unwrapped, key, value)
    return env


# ------------------------------------------------------------------ fake dm_control / dm_env
class StepType(enum.IntEnum):
    FIRST = 0
    MID = 1
    LAST = 2


class _TimeStep(namedtuple("TimeStep", ["step_type", "reward", "discount", "observation"])):
    def first(self):
        return self.step_type == StepType.FIRST

    def last(self):
        return self.step_type == StepType.LAST


class BoundedArray:
    def __init__(self, shape, dtype, minimum, maximum, name=None):
        self.shape, self.dtype, self.minimum, self.maximum, self.name = (
            tuple(shape),
            np.dtype(dtype),
            minimum,
            maximum,
            name,
        )


class _ShapelessSpec:
    """Observation spec without a 'shape' attribute (exercises the fallback in TimeStepToGymWrapper)."""


class FakePhysics:
    def __init__(self):
        self.render_calls = []

    def render(self, height, width, camera_id):
        self.render_calls.append((height, width, camera_id))
        return np.full((height, width, 3), 7 * camera_id, np.uint8)


class FakeDmEnv:
    """Minimal dm_control task: observation {'position': (2,), 'flag': scalar}, LAST after `length` steps."""

    def __init__(self, length=3, seed=None):
        self.length, self.seed, self.t = length, seed, 0
        self.physics = FakePhysics()
        self.received_actions = []

    def observation_spec(self):
        return {"position": BoundedArray((2,), np.float64, -1, 1), "flag": _ShapelessSpec()}

    def action_spec(self):
        return BoundedArray((2,), np.float64, -2.0, 2.0)

    def _observation(self):
        return {"position": np.full(2, float(self.t)), "flag": np.array([1.0])}

    def reset(self):
        self.t = 0
        return _TimeStep(StepType.FIRST, None, None, self._observation())

    def step(self, action):
        self.received_actions.append(np.asarray(action).copy())
        self.t += 1
        step_type = StepType.LAST if self.t >= self.length else StepType.MID
        return _TimeStep(step_type, 1.0, 0.5 if step_type == StepType.LAST else 1.0, self._observation())


class FakeActionScaleWrapper:
    """Rescales actions from [minimum, maximum] to the wrapped action spec."""

    def __init__(self, env, minimum, maximum):
        self._env, self.minimum, self.maximum = env, minimum, maximum

    def step(self, action):
        spec = self._env.action_spec()
        scale = (spec.maximum - spec.minimum) / (self.maximum - self.minimum)
        return self._env.step(spec.minimum + (np.asarray(action) - self.minimum) * scale)

    def action_spec(self):
        spec = self._env.action_spec()
        return BoundedArray(spec.shape, spec.dtype, self.minimum, self.maximum, "action")

    def observation_spec(self):
        return self._env.observation_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def install_fake_dm_control(
    monkeypatch, tasks=(("cartpole", "swingup"), ("ball_in_cup", "catch"), ("quadruped", "walk"))
):
    """Install fake dm_env and dm_control modules; returns the list of FakeDmEnv instances loaded."""
    loaded = []

    dm_env = types.ModuleType("dm_env")
    dm_env.Environment = type("Environment", (), {})
    dm_env.StepType = StepType
    dm_env.specs = types.SimpleNamespace(BoundedArray=BoundedArray)

    suite = types.ModuleType("dm_control.suite")
    suite.ALL_TASKS = tuple(tasks)

    def load(domain, task, task_kwargs=None, visualize_reward=False):
        env = FakeDmEnv(seed=(task_kwargs or {}).get("random"))
        env.domain, env.task = domain, task
        loaded.append(env)
        return env

    suite.load = load
    action_scale = types.ModuleType("dm_control.suite.wrappers.action_scale")
    action_scale.Wrapper = FakeActionScaleWrapper
    wrappers = types.ModuleType("dm_control.suite.wrappers")
    wrappers.action_scale = action_scale
    suite.wrappers = wrappers
    dm_control = types.ModuleType("dm_control")
    dm_control.suite = suite

    for name, module in {
        "dm_env": dm_env,
        "dm_control": dm_control,
        "dm_control.suite": suite,
        "dm_control.suite.wrappers": wrappers,
        "dm_control.suite.wrappers.action_scale": action_scale,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "envs.from_dmcontrol", raising=False)
    return loaded


# ------------------------------------------------------------- fake gymnasium / tensordict
class FakeTensorDict(dict):
    def copy(self):
        return FakeTensorDict(self)


class FakeGymnasiumWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


class FakeGymnasiumEnv:
    """New-style (gymnasium) env returning dict observations with an image and a vector."""

    metadata = {}
    reward_range = (-1.0, 1.0)

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = gym.spaces.Box(0, 255, (3, 8, 8), np.uint8)
        self.received_actions = []

    def _obs(self):
        return {"image": np.full((8, 8, 3), 255, np.uint8), "state": np.arange(3, dtype=np.float32)}

    def reset(self):
        return self._obs(), {"reset": True}

    def step(self, action):
        self.received_actions.append(action)
        return self._obs(), 0.5, False, True, {"step": True}


def install_fake_gymnasium_and_tensordict(monkeypatch):
    gymnasium = types.ModuleType("gymnasium")
    gymnasium.Wrapper = FakeGymnasiumWrapper
    tensordict = types.ModuleType("tensordict")
    tensordict.TensorDict = FakeTensorDict
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "tensordict", tensordict)
    monkeypatch.delitem(sys.modules, "envs.wrappers.to_tensordict", raising=False)


def install_fake_survival_suite(monkeypatch):
    install_fake_gymnasium_and_tensordict(monkeypatch)
    module = types.ModuleType("envs.from_survival")
    module.calls = []

    def make_env(name, image_size, max_episode_steps, action_repeat, seed=0):
        module.calls.append((name, image_size, max_episode_steps, action_repeat, seed))
        return FakeGymnasiumEnv()

    module.make_env = make_env
    monkeypatch.setitem(sys.modules, "envs.from_survival", module)
    return module


# -------------------------------------------------------------------------------- fake wandb
def install_fake_wandb(monkeypatch):
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)  # Libraries probe it with find_spec.
    wandb.calls = []
    wandb.init = lambda **kwargs: wandb.calls.append(("init", kwargs))
    wandb.finish = lambda: wandb.calls.append(("finish", {}))
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return wandb
