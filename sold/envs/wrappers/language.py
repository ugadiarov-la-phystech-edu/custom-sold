"""Streamed text instructions (multi_object_fetch.language) as a per-step memory of the instruction heard so far."""

import gym
import numpy as np
from typing import Any

PAD = 0  # T5's pad id: the stream emits it between repeats of the instruction.
REVEALS = ("stream", "full")


class LanguageMemory(gym.Wrapper):
    """Adds "tokens" to the observation: the token ids of the instruction heard so far (memory_length ids, 0-padded).

    Wraps a multi_object_fetch LanguageWrapper, which streams the episode's instruction one token per step ("token")
    and repeats it every repeat_task_every steps. reveal="stream" writes each token at the next position and starts
    over at position 0 after a pad, so a repeat rewrites the same ids; reveal="full" (an oracle) holds the whole
    instruction from the first step on. Each reset checks that the instruction fits and, when streamed, that a pad
    follows each repeat.
    """

    def __init__(self, env: gym.Env, memory_length: int, reveal: str = "stream") -> None:
        if reveal not in REVEALS:
            raise ValueError(f"Invalid reveal {reveal!r}; expected one of {REVEALS}.")
        super().__init__(env)
        self.memory_length = memory_length
        self.reveal = reveal
        self.memory = np.zeros(memory_length, np.int64)
        self.position = 0  # Where the next streamed token goes.
        tokens_space = gym.spaces.Box(0, np.iinfo(np.int32).max, (memory_length,), np.int64)
        self.observation_space = gym.spaces.Dict({**env.observation_space.spaces, "tokens": tokens_space})

    def reset(self) -> dict[str, Any]:
        obs = self.env.reset()
        instruction = np.atleast_1d(self.env.token_cache[self.env.instruction])
        self._check(instruction)
        self.memory[:], self.position = PAD, 0
        if self.reveal == "full":
            self.memory[: len(instruction)] = instruction
        else:
            self._hear(obs["token"])
        return self._observation(obs)

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        if self.reveal == "stream":
            self._hear(obs["token"])
        return self._observation(obs), reward, done, info

    def _check(self, instruction: np.ndarray) -> None:
        text, length, memory_length = self.env.instruction, len(instruction), self.memory_length
        if length > memory_length:
            raise ValueError(f"The instruction {text!r} has {length} tokens, more than memory_length={memory_length}.")
        repeat = self.env.repeat_task_every
        if self.reveal == "stream" and 0 < repeat <= length:
            raise ValueError(
                f"repeat_task_every={repeat} must exceed the {length} tokens of {text!r}: a pad must end each repeat."
            )

    def _hear(self, token: Any) -> None:
        if int(token) == PAD:
            self.position = 0
        else:
            self.memory[self.position] = int(token)
            self.position += 1

    def _observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {**obs, "tokens": self.memory.copy()}


def language_wrapper_class() -> type:
    """multi_object_fetch's LanguageWrapper, imported only when a task streams instructions."""
    from multi_object_fetch.language import LanguageWrapper

    return LanguageWrapper


def add_language(env: gym.Env, memory_length: int, repeat_task_every: int = 20, reveal: str = "stream") -> gym.Env:
    """env (multi_object_fetch: a Dict observation space and an instruction) with its instruction streamed one token
    per step, without drawing it on the frames, and remembered by a LanguageMemory."""
    env = language_wrapper_class()(env, repeat_task_every=repeat_task_every, visualize=False)
    return LanguageMemory(env, memory_length, reveal)
