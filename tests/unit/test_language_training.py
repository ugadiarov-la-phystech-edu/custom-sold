"""SOLD with instruction memories: the heads read them in training, imagination and action selection."""

from pathlib import Path
import pytest
from sold.data.ring_buffer import RingBufferDataset
from sold.train_sold import SOLDModule
from sold.utils.training import _autocast
from tests import fakes
from tests.conftest import (
    LANGUAGE_ENV,
    add_episode,
    fit_sold,
    make_episode,
    make_tiny_language_sold,
    make_tiny_sold,
    read_metrics,
)
from tests.golden import references
from tests.unit.test_compile import _assert_same_training, _steps
from tests.unit.test_training import _buffer_checkpoint
import torch

pytestmark = pytest.mark.usefixtures("fake_language")

RED, BLUE = (list(ids) + [0] * (8 - len(ids)) for ids in fakes.INSTRUCTIONS.values())


def _step_tokens(sold, batch_size=2):
    """Instruction memories that tell the steps apart: step t of every sample holds the id 100 + t."""
    tokens = torch.zeros(batch_size, sold.sequence_length, 8, dtype=torch.long)
    tokens[:, :, 0] = 100 + torch.arange(sold.sequence_length)
    return tokens


def _batch(sold):
    images, actions, rewards = references._tiny_sold_batch(sold)
    return {"obs": images * 255, "action": actions, "reward": rewards, "tokens": _step_tokens(sold)}


def _record_tokens(module):
    """The tokens module is called with, recorded from here on."""
    calls = []
    module.register_forward_pre_hook(lambda _, args, kwargs: calls.append(kwargs.get("tokens")), with_kwargs=True)
    return calls


def _steps_of(calls):
    """The steps each call's tokens come from (tokens of _step_tokens)."""
    return [(tokens[0, :, 0] - 100).tolist() for tokens in calls]


class LogRecorder:
    def __init__(self):
        self.names = []

    def __call__(self, name, *args, **kwargs):
        self.names.append(name)


# ---------------------------------------------------------------------------------- construction
def test_only_the_heads_read_the_instruction():
    sold = make_tiny_language_sold()
    assert sold.language == {"embed_dim": 4} and sold.hparams["language"] == {"embed_dim": 4}
    for head in (sold.actor, sold.critic, sold.critic_target, sold.reward_predictor):
        assert head.language_encoder is not None and head.language_encoder.embedding.embedding_dim == 4
    names = set(sold.state_dict())
    assert {"actor.language_encoder.embedding.weight", "critic_target.language_encoder.projection.bias"} <= names
    assert not any("language" in name for name in names if name.startswith(("autoencoder", "dynamics_predictor")))
    assert make_tiny_sold().language is None and not any("language" in name for name in make_tiny_sold().state_dict())


# ---------------------------------------------------------------------------------- training
def test_each_head_reads_the_memories_of_the_steps_it_scores():
    sold = make_tiny_language_sold()  # 2 context frames, 3 imagined steps: windows of 5 steps.
    sold.optimizers = sold.configure_optimizers()
    actor, critic = _record_tokens(sold.actor), _record_tokens(sold.critic)
    reward, target = _record_tokens(sold.reward_predictor), _record_tokens(sold.critic_target)
    outputs = sold.training_step(_batch(sold))
    assert all(torch.isfinite(outputs[key]) for key in ("reward_loss", "actor_loss", "critic_loss"))
    window = [0, 1, 2, 3, 4]
    assert _steps_of(reward) == [window, window]  # The reward loss, then imagination's rewards.
    assert _steps_of(actor) == [[0, 1], [0, 1, 2], [0, 1, 2, 3]]  # One action per imagined step.
    assert _steps_of(critic) == [window, window[:-1]]  # Imagination's values, then the critic loss.
    assert _steps_of(target) == [window[:-1]]


def test_imagination_tokens_from_every_start_hold_the_last_memory():
    sold = make_tiny_language_sold(start_imagination_from_every=True)
    tokens = _step_tokens(sold, batch_size=1)
    steps = (sold.imagination_tokens(tokens, num_context=2)[:, :, 0] - 100).tolist()
    assert steps == [[0, 1, 2, 3, 4], [1, 2, 3, 4, 4], [2, 3, 4, 4, 4], [3, 4, 4, 4, 4]]  # One rollout per start.
    assert torch.equal(make_tiny_language_sold().imagination_tokens(tokens, num_context=2), tokens)


@pytest.mark.parametrize("from_every", [False, True])
def test_imagine_ahead_carries_the_tokens_of_its_rollouts(from_every):
    sold = make_tiny_language_sold(start_imagination_from_every=from_every)
    batch = _batch(sold)
    slots = sold.autoencoder(batch["obs"] / 255, batch["action"][:, 1:])["slots"].detach()
    imagination = sold.imagine_ahead(slots, batch["action"], batch["tokens"])
    assert imagination.tokens.shape == (*imagination.imagined_slots.shape[:2], 8)
    assert make_tiny_sold().imagine_ahead(slots, batch["action"]).tokens is None


def test_language_figures_use_the_tokens():
    sold = make_tiny_language_sold()
    sold.optimizers = sold.configure_optimizers()
    sold.log = sold.log_image = recorder = LogRecorder()
    reward, actor = _record_tokens(sold.reward_predictor), _record_tokens(sold.actor)
    sold.training_step(_batch(sold), visualize=True)
    assert "reward_predictor_attention" in recorder.names and "actor_attention" in recorder.names
    assert all(tokens is not None for tokens in reward + actor)


def test_tokens_must_come_with_language_and_only_with_it():
    language, plain = make_tiny_language_sold(), make_tiny_sold()
    for sold in (language, plain):
        sold.optimizers = sold.configure_optimizers()
    batch = _batch(language)
    with pytest.raises(ValueError, match="The replay buffer has no instruction memories"):
        language.training_step({key: value for key, value in batch.items() if key != "tokens"})
    with pytest.raises(ValueError, match="The replay buffer has instruction memories .* without language"):
        plain.training_step(batch)
    image = torch.zeros(1, 3, 16, 16, dtype=torch.uint8)
    with pytest.raises(ValueError, match="The observation has no instruction memories"):
        language.select_action(image, is_first=True)
    with pytest.raises(ValueError, match="The observation has instruction memories"):
        plain.select_action({"image": image, "tokens": torch.zeros(1, 8, dtype=torch.long)}, is_first=True)


# ---------------------------------------------------------------------------------- action selection
def _observations(num_steps=4, num_envs=2):
    """Images and instruction memories of num_steps steps: RED streamed into environment 0, BLUE into 1."""
    images = torch.randint(0, 255, (num_steps, num_envs, 3, 16, 16), dtype=torch.uint8)
    tokens = torch.zeros(num_steps, num_envs, 8, dtype=torch.long)
    for t in range(num_steps):
        for env, ids in enumerate([RED, BLUE][:num_envs]):
            tokens[t, env, : t + 1] = torch.tensor(ids[: t + 1])
    return [{"image": images[t], "tokens": tokens[t]} for t in range(num_steps)]


def test_select_action_keeps_a_token_history_next_to_the_slot_history():
    sold = make_tiny_language_sold(fakes.FakeBatchEnv(2, language=LANGUAGE_ENV))
    observations = _observations()
    for t, obs in enumerate(observations):
        action = sold.select_action(obs, is_first=t == 0, mode="eval")
        history = sold._token_history
        assert history.shape == (2, 8, 8) and torch.equal(history[:, t], obs["tokens"])
        assert history[:, t + 1 :].sum() == 0
        grown = sold.actor(sold._slot_history[:, : t + 1], start=t, tokens=history[:, : t + 1])
        assert torch.allclose(action, grown.mode[:, 0].clamp(-1, 1), atol=1e-5)  # As over the growing history.
    sold.select_action(observations[0], is_first=True, mode="random")
    assert sold._step == 1 and sold._token_history[:, 1:].sum() == 0  # Reset at the next episode.


def test_actions_depend_on_the_instruction():
    sold = make_tiny_language_sold()
    obs = _observations(num_envs=1)[3]
    red = sold.select_action(obs, is_first=True, mode="eval")
    blue = sold.select_action({"image": obs["image"], "tokens": torch.tensor([BLUE])}, is_first=True, mode="eval")
    assert not torch.allclose(red, blue)


# ---------------------------------------------------------------------------------- training runs
def _assert_streamed_memories(tokens, boundaries):
    """Each stored episode's memories grow token by token into one of the instructions, then stay."""
    for start, end in boundaries:
        episode = tokens[start:end].tolist()
        final = episode[-1]
        assert final in (RED, BLUE)
        length = sum(1 for token in final if token)
        assert episode == [final[: t + 1] + [0] * (7 - t) for t in range(length)] + [final] * (end - start - length)


def test_fit_stores_the_instruction_memories_and_resumes(tmp_path):
    torch.manual_seed(0)
    sold = make_tiny_language_sold(save_replay_buffer=True)
    logger = fit_sold(sold, tmp_path / "first")
    buffer = sold.replay_buffer
    assert buffer.ring_buffer["tokens"].dtype.name == "int64" and buffer.ring_buffer["tokens"].shape == (300, 8)
    _assert_streamed_memories(buffer.ring_buffer["tokens"], buffer.episode_boundaries)
    assert any("eval/episode_return" in record for record in read_metrics(logger.log_dir))
    checkpoint = sorted(Path(logger.log_dir).glob("checkpoints/*.ckpt"))[-1]
    assert torch.load(checkpoint, weights_only=False)["hyper_parameters"]["language"] == {"embed_dim": 4}
    resumed = make_tiny_language_sold(max_steps=60)
    fit_sold(resumed, tmp_path / "resumed", checkpoint=checkpoint)
    assert resumed.num_steps > sold.num_steps and "tokens" in resumed.replay_buffer.ring_buffer
    loaded = SOLDModule.load(checkpoint, env=fakes.make_fake_env(language=LANGUAGE_ENV))
    assert loaded.language == {"embed_dim": 4} and loaded.actor.language_encoder is not None


def test_fit_with_several_environments_and_language(tmp_path):
    sold = make_tiny_language_sold(fakes.FakeBatchEnv(2, language=LANGUAGE_ENV), save_replay_buffer=False)
    fit_sold(sold, tmp_path)
    buffer = sold.replay_buffer
    assert buffer.num_episodes >= 2
    _assert_streamed_memories(buffer.ring_buffer["tokens"], buffer.episode_boundaries)


def _saved_buffer(tmp_path, with_tokens):
    saved = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "saved"))
    episode = make_episode(5)
    if with_tokens:
        episode["tokens"] = [torch.tensor(RED) for _ in range(5)]
    add_episode(saved, episode)
    saved.save_bookkeeping()
    return saved


@pytest.mark.parametrize("language", [True, False])
def test_resuming_needs_a_replay_buffer_with_tokens_exactly_with_language(language, tmp_path):
    sold = make_tiny_language_sold() if language else make_tiny_sold()
    sold.replay_buffer = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "run"))
    with pytest.raises(ValueError, match="The saved replay buffer has (no )?instruction memories"):
        sold.restore(_buffer_checkpoint(sold, _saved_buffer(tmp_path / "other", with_tokens=not language)))
    sold.restore(_buffer_checkpoint(sold, _saved_buffer(tmp_path / "same", with_tokens=language)))
    assert sold.replay_buffer.num_episodes == 1 and ("tokens" in sold.replay_buffer.ring_buffer) == language


def test_played_episodes_record_the_images():
    sold = make_tiny_language_sold()
    episode = sold.runner.play_episode(mode="eval")
    assert len(episode["obs"]) == 9 and all(obs.shape == (3, 16, 16) for obs in episode["obs"])


# ---------------------------------------------------------------------------------- compiling
@pytest.mark.usefixtures("eager_randomness")
def test_compiled_language_stages_train_like_eager_ones(monkeypatch, aot_eager):
    torch.manual_seed(0)
    eager = make_tiny_language_sold()
    torch.manual_seed(0)
    compiled = make_tiny_language_sold()
    compiled.compile_stages("default")
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    tokens = _step_tokens(eager)
    _assert_same_training(_steps(compiled, 3, tokens=tokens), _steps(eager, 3, tokens=tokens))


def _two_language_models(device="cpu"):
    models = []
    for _ in range(2):
        torch.manual_seed(0)
        models.append(make_tiny_language_sold(fakes.FakeBatchEnv(2, language=LANGUAGE_ENV)).to(device))
    return models


def _actions(sold, observations, mode, episodes=1):
    actions = []
    for _ in range(episodes):
        torch.manual_seed(0)
        for t, obs in enumerate(observations):
            actions.append(sold._policy(obs, is_first=t == 0, mode=mode).clone())
    return actions


@pytest.mark.usefixtures("eager_randomness")
@pytest.mark.parametrize("mode", ["eval", "train"])
def test_compiled_action_selection_with_tokens_equals_eager(mode, monkeypatch, aot_eager):
    eager, compiled = _two_language_models()
    compiled.compile_stages("default", ["encode", "act"])
    observations = _observations(num_steps=8)
    expected = _actions(eager, observations, mode)
    _actions(compiled, observations, mode)  # Compiles.
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    for actual, reference in zip(_actions(compiled, observations, mode), expected):
        assert torch.allclose(actual, reference, atol=1e-5)


def test_autocast_passes_non_tensor_outputs_through():
    outputs = {"loss": torch.ones(2, dtype=torch.bfloat16), "tokens": None, "ids": torch.ones(2).long()}
    outputs = _autocast(lambda: outputs, "cpu")()
    assert outputs["loss"].dtype == torch.float32 and outputs["tokens"] is None and outputs["ids"].dtype == torch.long


@pytest.mark.usefixtures("eager_randomness")
def test_bfloat16_language_training_stays_close_to_float32():
    torch.manual_seed(0)
    float32 = make_tiny_language_sold()
    torch.manual_seed(0)
    bfloat16 = make_tiny_language_sold()
    bfloat16.autocast_stages("bfloat16", ["dynamics_loss", "reward_loss", "imagination", "actor_loss", "critic_loss"])
    tokens = _step_tokens(float32)
    _assert_same_training(_steps(bfloat16, 3, tokens=tokens), _steps(float32, 3, tokens=tokens), rtol=0.1, atol=0.02)
    assert all(p.dtype == torch.float32 for p in bfloat16.actor.language_encoder.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.usefixtures("eager_randomness")
def test_cuda_graphs_with_language_train_and_act_like_eager_stages():
    from torch._dynamo.utils import counters

    eager, graphed = _two_language_models("cuda")
    graphed.fused_optimizers = eager.fused_optimizers = True
    graphed.compile_stages("reduce-overhead")
    tokens = _step_tokens(eager)
    # Four steps: compiled code warms up, records the CUDA graphs, then replays them.
    _assert_same_training(_steps(graphed, 4, "cuda", tokens), _steps(eager, 4, "cuda", tokens))
    observations = [{key: value.cuda() for key, value in obs.items()} for obs in _observations(num_steps=8)]
    expected = _actions(eager, observations, "eval", 3)
    for actual, reference in zip(_actions(graphed, observations, "eval", 3), expected):
        assert torch.allclose(actual, reference, atol=1e-4)
    assert counters["inductor"]["cudagraph_skips"] == 0
