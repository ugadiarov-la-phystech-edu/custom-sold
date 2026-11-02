"""envs/vector.py: batches of environments."""

import functools
import multiprocessing
import numpy as np
import pytest
import random
import signal
from sold.envs.vector import SingleEnv, VectorEnv, env_worker
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


def test_vector_env_steps_each_environment_in_its_own_process():
    env = VectorEnv(functools.partial(fakes.make_seeded_fake_env, max_episode_steps=4), num_envs=2, seed=3)
    try:
        assert env.num_envs == 2 and env.max_episode_steps == 4 and env.action_space.shape == (2,)
        assert len({process.pid for process in env.processes}) == 2
        obs = env.reset()
        assert obs.shape == (2, 3, 16, 16) and obs.dtype == torch.uint8
        assert [obs[index, 1].max().item() for index in range(2)] == [3, 1003 % 256]  # Seeds seed + 1000 * index.
        for _ in range(4):
            obs, rewards, dones, infos = env.step(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        assert dones == [True, True] and len(rewards) == 2 and "success" in infos[1]
        assert not torch.equal(obs[0, 0], obs[1, 0])  # Each environment took its own action.
    finally:
        env.close()
    assert not any(process.is_alive() for process in env.processes)
    env.close()  # Closing again does nothing.


def test_vector_env_raises_the_errors_of_its_workers():
    with pytest.raises(RuntimeError, match="(?s)Environment worker 0 failed.*broken environment"):
        VectorEnv(fakes.make_broken_env, num_envs=1)


def test_vector_env_terminates_workers_that_do_not_end():
    class StuckProcess:
        terminated = False

        def join(self, timeout):
            self.timeout = timeout

        def is_alive(self):
            return not self.terminated

        def terminate(self):
            self.terminated = True

    env = object.__new__(VectorEnv)
    env.closed, env.connections, env.processes = False, [], [StuckProcess()]
    env.close()
    assert env.processes[0].timeout == 10 and env.processes[0].terminated


def test_env_worker_serves_requests(monkeypatch):
    """The worker loop run in this process (coverage does not see the spawned workers)."""
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    random_state, numpy_state = random.getstate(), np.random.get_state()  # The worker seeds both.
    seeds = []

    def factory(seed):
        seeds.append(seed)
        return fakes.make_fake_env(seed=seed)

    connection, child = multiprocessing.Pipe()
    for request in [("reset", None), ("step", np.zeros(2, np.float32)), ("close", None)]:
        connection.send(request)
    env_worker(child, factory, 7)
    replies = [connection.recv() for _ in range(4)]
    assert seeds == [7] and [status for status, _ in replies] == ["ok"] * 4
    assert replies[0][1][1] == 8 and replies[1][1].shape == (3, 16, 16) and replies[2][1][2] is False

    connection, child = multiprocessing.Pipe()
    connection.send(("render", None))
    env_worker(child, factory, 7)
    connection.recv()
    status, message = connection.recv()
    assert status == "error" and "Unknown command: render" in message
    random.setstate(random_state)
    np.random.set_state(numpy_state)
