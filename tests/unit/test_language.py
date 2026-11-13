"""Language: the instruction memory of streamed instructions (sold.envs.wrappers.language)."""

import numpy as np
import pytest
from sold.envs.wrappers import language
from sold.envs.wrappers.language import LanguageMemory, add_language
from tests import fakes

RED, BLUE = (list(ids) for ids in fakes.INSTRUCTIONS.values())  # Episodes alternate between them.


def memory_env(memory_length=8, reveal="stream", repeat_task_every=20, episode_length=50):
    inner = fakes.FakeLanguageWrapper(fakes.FakePixelEnv(episode_length), repeat_task_every=repeat_task_every)
    return LanguageMemory(inner, memory_length, reveal)


def padded(ids, length=8):
    return ids + [0] * (length - len(ids))


def episode_tokens(env, num_steps):
    """The memory after the reset and after each of num_steps steps."""
    memories = [env.reset()["tokens"]]
    for _ in range(num_steps):
        memories.append(env.step(np.zeros(2))[0]["tokens"])
    return [memory.tolist() for memory in memories]


def test_streamed_tokens_are_written_by_position():
    memories = episode_tokens(memory_env(), num_steps=6)
    assert memories[:4] == [padded(RED[: step + 1]) for step in range(4)]
    assert memories[4:] == [padded(RED)] * 3  # The pads after the instruction keep it.


def test_repeats_rewrite_the_same_ids():
    env = memory_env(repeat_task_every=10)
    memories = episode_tokens(env, num_steps=28)
    assert all(memory == padded(RED) for memory in memories[3:])  # Repeats at steps 10 and 20.
    assert env.position == 0  # The pads after the last repeat.


def test_reset_clears_the_memory_for_the_next_instruction():
    env = memory_env()
    episode_tokens(env, num_steps=10)
    assert env.reset()["tokens"].tolist() == padded(BLUE[:1])
    assert episode_tokens(env, num_steps=8)[-1] == padded(RED)  # The episode after BLUE is RED again.


def test_full_reveal_holds_the_whole_instruction_from_the_reset():
    env = memory_env(reveal="full")
    assert episode_tokens(env, num_steps=25) == [padded(RED)] * 26
    assert episode_tokens(env, num_steps=2) == [padded(BLUE)] * 3


def test_memory_adds_tokens_to_the_observation_and_its_space():
    env = memory_env(memory_length=6)
    obs = env.reset()
    assert set(obs) == {"observation", "token", "tokens"}
    assert obs["tokens"].dtype == np.int64 and obs["tokens"].shape == (6,)
    space = env.observation_space.spaces
    assert set(space) == {"observation", "token", "tokens"} and space["tokens"].shape == (6,)
    assert space["tokens"].contains(obs["tokens"])
    obs["tokens"][0] = 7  # Observations are copies of the memory.
    assert env.memory[0] == RED[0]


@pytest.mark.parametrize("reveal", ["stream", "full"])
def test_reset_rejects_an_instruction_longer_than_the_memory(reveal):
    with pytest.raises(ValueError, match="'reach the red object' has 4 tokens, more than memory_length=3"):
        memory_env(memory_length=3, reveal=reveal).reset()


@pytest.mark.parametrize("repeat_task_every", [1, 4])
def test_reset_rejects_streams_without_a_pad_between_repeats(repeat_task_every):
    with pytest.raises(ValueError, match=f"repeat_task_every={repeat_task_every} must exceed the 4 tokens"):
        memory_env(repeat_task_every=repeat_task_every).reset()


def test_streams_with_a_pad_between_repeats_or_without_repeats_are_accepted():
    assert episode_tokens(memory_env(repeat_task_every=5), num_steps=12)[-1] == padded(RED)
    assert episode_tokens(memory_env(repeat_task_every=0), num_steps=30)[-1] == padded(RED)
    assert episode_tokens(memory_env(repeat_task_every=2, reveal="full"), num_steps=3)[-1] == padded(RED)
    assert episode_tokens(memory_env(memory_length=24), num_steps=5)[-1] == padded(RED, 24)


def test_invalid_reveal_raises():
    with pytest.raises(ValueError, match="Invalid reveal 'all'"):
        memory_env(reveal="all")


def test_add_language_streams_without_overlay(fake_language):
    env = add_language(fakes.FakePixelEnv(), memory_length=8, repeat_task_every=12, reveal="full")
    assert isinstance(env, LanguageMemory) and env.reveal == "full" and env.memory_length == 8
    assert isinstance(env.env, fakes.FakeLanguageWrapper)
    assert env.env.visualize is False and env.env.repeat_task_every == 12
    assert add_language(fakes.FakePixelEnv(), memory_length=8).reveal == "stream"


def test_language_wrapper_class_is_multi_object_fetchs():
    pytest.importorskip("multi_object_fetch.language")
    from multi_object_fetch.language import LanguageWrapper

    assert language.language_wrapper_class() is LanguageWrapper
