"""One test per bug in BUGS_AND_PITFALLS.md (number in the test name).

Each test asserts the CORRECT behavior, so it fails until the bug is fixed. Run them with `pytest -m bug`;
exclude them with `pytest -m "not bug"` to use the rest of the suite as a refactoring safety net.
"""

import ast
import functools
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import envs
from tests import fakes
from tests.conftest import CONFIGS_DIR, REPO_DIR, SOLD_DIR, make_online_trainer, make_tiny_savi, make_tiny_sold
from tests.unit.test_datasets import _add, _episode

pytestmark = pytest.mark.bug


# ------------------------------------------------------------------------------------- #2
def test_02_savi_dataset_path_is_not_hardcoded():
    dataset = OmegaConf.load(CONFIGS_DIR / "train_autoencoder.yaml").dataset
    configured = SOLD_DIR / dataset.path / dataset.data_dir
    assert OmegaConf.is_missing(dataset, "data_dir") or configured.exists(), \
        f"dataset points at {configured}, which only exists on the author's machine"


# ------------------------------------------------------------------------------------ #3a
def test_03a_progress_bar_creates_its_display_on_train_start(tiny_sold):
    from lightning import Trainer
    from utils.training import OnlineProgressBar
    bar = OnlineProgressBar()
    trainer = Trainer(accelerator="cpu", logger=False, callbacks=[bar], enable_model_summary=False)
    bar.setup(trainer, tiny_sold, stage="fit")
    bar.on_train_start(trainer, tiny_sold)
    # Lightning >= 2.5.3 calls on_train_batch_end -> _add_task, which asserts that the display exists.
    assert bar.progress is not None


# ------------------------------------------------------------------------------------- #4
def test_04_resuming_restores_the_saved_replay_buffer(tiny_sold, tmp_path):
    from datasets.ring_buffer import RingBufferDataset
    saved = RingBufferDataset(20, 2, 3, save_path=str(tmp_path / "saved"))
    _add(saved, _episode(5))
    checkpoint = {"num_steps": 5, "num_episodes": 1, "replay_buffer_path": tmp_path / "saved",
                  "replay_buffer_capacity": 20, "replay_buffer_info": saved.info, "replay_buffer_head": saved.head,
                  "replay_buffer_tail": saved.tail, "replay_buffer_episode_boundaries": saved.episode_boundaries}
    with mock.patch.object(type(tiny_sold), "logger", new_callable=mock.PropertyMock,
                           return_value=SimpleNamespace(log_dir=str(tmp_path / "run"))):
        tiny_sold.on_load_checkpoint(checkpoint)
    assert tiny_sold.replay_buffer.num_episodes == 1 and tiny_sold.replay_buffer.tail == 5


# ------------------------------------------------------------------------------------- #5
def test_05_pruning_keeps_the_newest_replay_buffer(tiny_sold, tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    (checkpoints_dir / "replay_buffer-steps=9000-episode=180").mkdir(parents=True)
    live = tmp_path / "live"
    live.mkdir()
    (live / "obs.dat").write_bytes(b"x")
    tiny_sold.save_replay_buffer, tiny_sold.num_steps, tiny_sold.num_episodes = True, 10500, 210
    tiny_sold.replay_buffer = SimpleNamespace(save_path=str(live), capacity=1, info=None, head=0, tail=0,
                                              episode_boundaries=[])
    checkpoint = {}
    with mock.patch.object(type(tiny_sold), "checkpoints_dir", new_callable=mock.PropertyMock,
                           return_value=checkpoints_dir):
        tiny_sold.on_save_checkpoint(checkpoint)
    assert [p.name for p in checkpoints_dir.iterdir()] == ["replay_buffer-steps=10500-episode=210"]
    assert checkpoint["replay_buffer_path"].exists()


# ------------------------------------------------------------------------------------- #6
def test_06_latest_checkpoint_loads_without_a_trainer(tmp_path):
    from train_sold import SOLDModule
    sold = make_tiny_sold(save_replay_buffer=True, max_steps=30)  # Evaluates at steps 0 and 24.
    trainer = make_online_trainer(tmp_path)
    trainer.fit(sold)
    checkpoints_dir = Path(trainer.logger.log_dir) / "checkpoints"
    assert list(checkpoints_dir.glob("replay_buffer-*")), "precondition: the replay buffer copy exists"
    latest = sorted(checkpoints_dir.glob("*.ckpt"))[-1]
    loaded = SOLDModule.load_from_checkpoint(latest, env=fakes.make_fake_env())  # As evaluate_sold.py does.
    assert loaded.num_steps == torch.load(latest, weights_only=False)["num_steps"]


# ------------------------------------------------------------------------------------- #7
def test_07_evaluation_updates_the_last_action():
    import evaluate_sold
    sold = make_tiny_sold()
    seen = []
    original = sold.autoencoder.encode
    sold.autoencoder.encode = lambda images, actions, prior_slots=None: (seen.append(actions.clone()),
                                                                          original(images, actions, prior_slots))[1]
    evaluate_sold.play_episode(sold, mode="eval")
    assert not torch.isnan(seen[-1]).any(), "the SAVi encoder still receives the initial NaN action"


# ------------------------------------------------------------------------------------- #8
def test_08_survival_suite_is_usable_or_rejected_cleanly():
    try:
        envs.make_env("survival", "Forage", (16, 16), 10, 1)
    except ValueError:
        pass  # Removing the suite is a valid fix.


# ------------------------------------------------------------------------------------- #9
@pytest.mark.mujoco
def test_09_reach_episodes_last_max_episode_steps():
    env = envs.make_env("mof", "ReachRed_0to0Distractors_Dense-v1", (64, 64), max_episode_steps=50, action_repeat=2)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, done, _ = env.step(torch.zeros(env.action_space.shape))
        steps += 1
    assert steps == 50


# ------------------------------------------------------------------------------------ #11
@pytest.mark.mujoco
def test_11_the_seed_makes_environments_reproducible():
    first = [envs.make_env("mof", "PushRed_2to2Distractors_Dense-v1", (64, 64), 50, 2, seed=3).reset()
             for _ in range(2)]
    assert torch.equal(first[0], first[1])


# ------------------------------------------------------------------------------------ #12
def test_12_distractor_wrapper_does_not_leak_environments(monkeypatch):
    import envs.from_mof as from_mof
    from envs.wrappers.pixels import Pixels
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


# ------------------------------------------------------------------------------------ #13
def test_13_savi_pretraining_uses_the_action_that_led_to_the_next_frame(tmp_path):
    import generate_dataset
    from datasets.image import NPZDataset
    from train_autoencoder import AutoencoderModule
    cfg = OmegaConf.create({"save_format": "npz", "env": {"_target_": "tests.fakes.make_fake_env",
                                                          "max_episode_steps": 6, "action_repeat": 1}})
    (tmp_path / "run" / "val").mkdir(parents=True)
    generate_dataset.save_episode(str(tmp_path / "run" / "val" / "0"), cfg)
    taken = np.load(tmp_path / "run" / "val" / "0" / "episode.npz")["actions"]  # taken[t] moves frame t -> t+1.
    images, actions = NPZDataset(str(tmp_path), "run", "val", 4)[0]

    savi = make_tiny_savi(action_conditional=True)
    used = []
    original = savi.predictor.forward
    savi.predictor.forward = lambda slots, actions: (used.append(actions[0].clone()), original(slots, actions))[1]
    AutoencoderModule(savi, functools.partial(torch.optim.Adam)).compute_reconstruction_loss(images[None], actions[None])
    for t in range(3):  # The predictor turns the slots of frame t into a prediction for frame t + 1.
        assert torch.allclose(used[t], torch.from_numpy(taken[t])), f"step {t} uses the action of step {t + 1}"


# ------------------------------------------------------------------------------------ #14
def test_14_episode_of_exactly_sequence_length_can_be_loaded(tmp_path):
    from datasets.image import load_image_dataset
    from tests.conftest import write_image_dataset
    write_image_dataset(tmp_path / "run", "npz", episode_length=4)
    images, _ = load_image_dataset(str(tmp_path), "run", "train", sequence_length=4)[0]
    assert images.shape[0] == 4


def test_14_random_crop_can_use_the_last_window(tmp_path):
    from datasets.image import load_image_dataset
    from tests.conftest import write_image_dataset
    write_image_dataset(tmp_path / "run", "npz", episode_length=7)
    dataset = load_image_dataset(str(tmp_path), "run", "train", sequence_length=4)
    last_window = np.load(tmp_path / "run" / "train" / "0" / "episode.npz")["actions"][3:7]
    assert any(len(dataset[0][1]) == len(last_window) and np.allclose(dataset[0][1].numpy(), last_window)
               for _ in range(200))


# ------------------------------------------------------------------------------------ #15
def test_15_pixels_wrapper_does_not_require_multi_object_fetch(monkeypatch):
    monkeypatch.setitem(sys.modules, "multi_object_fetch", None)
    monkeypatch.setitem(sys.modules, "multi_object_fetch.env", None)
    monkeypatch.delitem(sys.modules, "envs.wrappers.pixels")
    importlib.import_module("envs.wrappers.pixels")


# ------------------------------------------------------------------------------------ #16
def test_16_all_scripts_use_the_same_hydra_version_base():
    version_bases = {}
    for script in ["generate_dataset.py", "train_autoencoder.py", "train_sold.py", "evaluate_sold.py"]:
        tree = ast.parse((SOLD_DIR / script).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "main":
                keywords = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
                version_bases[script] = keywords.get("version_base", "<not set>")
    assert len(set(version_bases.values())) == 1, version_bases


# ------------------------------------------------------------------------------------ #17
def test_17_savi_initializes_slot_distribution_parameters():
    from modeling.autoencoder.savi import Corrector

    class CorrectorWithSlotDistribution(Corrector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.slots_mu = torch.nn.Parameter(torch.zeros(1, 1, self.slot_dim))
            self.slots_sigma = torch.nn.Parameter(torch.zeros(1, 1, self.slot_dim))

    savi = make_tiny_savi()
    savi.corrector = CorrectorWithSlotDistribution(2, 8, 8, 1, 2, 8)
    savi._initialize_parameters()  # Reads the non-existent corrector.dim_slots.


def test_17_actions_are_clamped_per_dimension():
    env = fakes.make_fake_env()
    env.unwrapped.action_space = env.action_space = type(env.action_space)(
        low=np.array([-1.0, -0.1], np.float32), high=np.array([1.0, 0.1], np.float32))
    sold = make_tiny_sold(env)
    obs = torch.zeros(3, 16, 16, dtype=torch.uint8)
    actions = torch.stack([sold.select_action(obs, is_first=True, mode="train") for _ in range(50)])
    assert actions[:, 1].abs().max() <= 0.1 + 1e-6


def test_17_autoencoder_optimizer_state_is_checkpointed(tmp_path):
    sold = make_tiny_sold(finetune_autoencoder=True, max_steps=20)
    trainer = make_online_trainer(tmp_path)
    trainer.fit(sold)
    checkpoint = torch.load(next((Path(trainer.logger.log_dir) / "checkpoints").glob("*.ckpt")), weights_only=False)
    saved = sum(len(group["params"]) for state in checkpoint["optimizer_states"] for group in state["param_groups"])
    expected = sum(len(list(module.parameters())) for module in
                   [sold.dynamics_predictor, sold.reward_predictor, sold.actor, sold.critic, sold.autoencoder])
    assert saved == expected


def test_17_no_autoencoder_graph_without_finetuning(tmp_path, monkeypatch):
    from train_autoencoder import AutoencoderModule
    recorded = []
    original = AutoencoderModule.compute_reconstruction_loss

    def recording(module, images, actions):
        outputs = original(module, images, actions)
        recorded.append(outputs["reconstruction_loss"].requires_grad)
        return outputs

    monkeypatch.setattr(AutoencoderModule, "compute_reconstruction_loss", recording)
    trainer = make_online_trainer(tmp_path)
    trainer.fit(make_tiny_sold(finetune_autoencoder=False, max_steps=15))
    assert recorded and not any(recorded)


# ------------------------------------------------------------------------------------ #18
def test_18_last_metrics_record_is_written_when_training_ends(tmp_path):
    import json
    from utils.logging import ExtendedTensorBoardLogger
    logger = ExtendedTensorBoardLogger(save_dir=str(tmp_path), name="logs")
    logger.log_metrics({"loss": 1.0}, step=1)
    logger.log_metrics({"loss": 2.0}, step=2)
    logger.finalize("success")
    records = [json.loads(line) for line in open(Path(logger.log_dir) / "metrics" / "metrics.jsonl")]
    assert records[-1] == {"step": 2, "loss": 2.0}


# ------------------------------------------------------------------------------------ #19
def test_19_replay_buffer_can_sample_the_last_window(tmp_path):
    from datasets.ring_buffer import RingBufferDataset
    buffer = RingBufferDataset(capacity=10, batch_size=64, sequence_length=3, save_path=str(tmp_path))
    _add(buffer, _episode(4))
    starts = set(buffer.sample()["obs"][:, 0, 0, 0, 0].tolist())
    assert starts == {0, 1}


# ------------------------------------------------------------------------------------ #20
def test_20_wrong_mask_size_raises_value_error():
    from modeling.attention_mask import CausalMask
    mask = CausalMask(2, 1, 1)
    mask._full_mask = torch.zeros(1, 3, 3)
    with pytest.raises(ValueError):
        mask(batch_size=1, sequence_length=1)


# ------------------------------------------------------------------------------------ #21
def test_21_mask_plot_title_shows_the_requested_head(monkeypatch):
    import matplotlib.pyplot as plt
    from modeling.attention_mask import CausalMask
    monkeypatch.setattr(plt, "show", lambda: None)
    CausalMask(2, 2, 1).show(head=1)
    title = plt.gca().get_title()
    plt.close("all")
    assert title.endswith("(head=1)")


# ------------------------------------------------------------------------------------ #22
def test_22_attention_visualization_does_not_stack_forward_wrappers():
    from modeling.sold.prediction import TwoHotPredictor
    from utils.visualizations import get_attention_weights
    model = TwoHotPredictor(max_episode_steps=4, num_slots=2, slot_dim=4, token_dim=8, num_heads=2, num_layers=1,
                            hidden_dim=8)
    slots = torch.randn(1, 3, 2, 4)
    for _ in range(sys.getrecursionlimit() + 100):  # One call per evaluation during training.
        get_attention_weights(model, slots)
    model(slots)  # RecursionError while the wrappers pile up.
    assert all("forward" not in vars(m) for m in model.modules() if isinstance(m, torch.nn.MultiheadAttention))


# ------------------------------------------------------------------------------------ #23
def test_23_cnn_autoencoder_config_can_be_instantiated():
    cfg = OmegaConf.load(CONFIGS_DIR / "autoencoder" / "cnn.yaml")
    cfg.encoder.image_size = cfg.decoder.image_size = [16, 16]
    autoencoder = hydra.utils.instantiate(cfg)
    assert autoencoder(torch.rand(1, 2, 3, 16, 16), torch.zeros(1, 1, 2))["reconstructions"].shape == (1, 2, 3, 16, 16)


# ------------------------------------------------------------------------------------ #24
def test_24_pixels_observation_matches_observation_space_for_non_square_sizes():
    from envs.wrappers.pixels import Pixels
    env = Pixels(fakes.FakePixelEnv(), image_size=(8, 6))
    assert env.reset().shape == env.observation_space.shape


# ------------------------------------------------------------------------------------ #25
def test_25_ring_buffer_accepts_predefined_fields(tmp_path):
    from datasets.info import EpisodeDatasetInfo, Fields, TensorType
    from datasets.ring_buffer import RingBufferDataset
    fields = Fields(obs=TensorType(device="cpu", dtype=torch.uint8, shape=(3, 2, 2)))
    buffer = RingBufferDataset(5, 1, 1, info=EpisodeDatasetInfo(fields=fields), save_path=str(tmp_path))
    assert buffer.ring_buffer["obs"].shape == (5, 3, 2, 2)


# ------------------------------------------------------------------------------------ #26
def test_26_entropy_bonus_uses_the_discount_of_its_own_step():
    sold = make_tiny_sold()
    sold.after_eval = False
    images = torch.rand(2, sold.sequence_length, 3, 16, 16)
    actions = torch.rand(2, sold.sequence_length, 2)
    slots = sold.autoencoder(images, actions[:, 1:])["slots"].detach()
    returns, values_targ, _, log_probs, entropies = sold.imagine_ahead(slots, actions)
    assert entropies.shape == (2, 3, 1)  # As produced by imagination: one entropy per (sample, step).
    entropies = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).unsqueeze(-1)
    losses = sold.compute_actor_loss(returns, values_targ, log_probs, entropies)
    expected = -sold.actor_entropy_loss_weight * torch.mean(sold.discounts * entropies.squeeze(-1))
    assert torch.allclose(losses["actor_entropy_loss"], expected), "each entropy is weighted by the mean discount"


# ------------------------------------------------------------------------------------ #27
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU to create a GPU checkpoint")
def test_27_savi_checkpoints_saved_on_gpu_load_on_cpu(tmp_path):
    from lightning import Trainer
    from train_autoencoder import AutoencoderModule, load_autoencoder
    module = AutoencoderModule(make_tiny_savi().cuda(), functools.partial(torch.optim.Adam))
    trainer = Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, max_steps=0)
    trainer.strategy.connect(module)
    trainer.save_checkpoint(tmp_path / "savi.ckpt")
    autoencoder = load_autoencoder(str(tmp_path / "savi.ckpt"))
    assert next(autoencoder.parameters()).device.type == "cpu", "a CPU-only machine could not load this checkpoint"
