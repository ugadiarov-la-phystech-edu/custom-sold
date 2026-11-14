"""Run the four Hydra entry points in-process with the real configs, a fake environment and toy model sizes."""

from hydra.core.global_hydra import GlobalHydra
import json
import math
import os
import pytest
from sold import evaluate_sold, generate_dataset, train_autoencoder, train_sold
import sys
from tests import fakes
from tests.conftest import make_tiny_savi, read_metrics
import torch

pytestmark = pytest.mark.slow

# make_fake_env ignores 'suite' and 'name' (kept because print_summary reads the name).
FAKE_ENV = ["env._target_=tests.fakes.make_fake_env", "env.image_size=[16,16]", "env.action_repeat=1"]
PREDICTOR = ["token_dim=8", "num_heads=2", "num_layers=1", "hidden_dim=16"]


def run_entry_point(monkeypatch, main, overrides):
    GlobalHydra.instance().clear()
    monkeypatch.setattr(sys, "argv", [main.__module__ + ".py", *overrides])
    # Hydra resolves config_path="../configs" relative to the script file only when it runs as __main__.
    monkeypatch.setattr(main.__wrapped__, "__module__", "__main__")
    precision = torch.get_float32_matmul_precision()
    try:
        main()
        torch.get_float32_matmul_precision()  # Lightning calls it on GPUs; it raises if the TF32 APIs are mixed.
    finally:
        GlobalHydra.instance().clear()
        torch.set_float32_matmul_precision(precision)  # train_sold sets it for the whole process.


def sold_overrides(run_dir, savi_checkpoint, **values):
    overrides = [
        f"hydra.run.dir={run_dir}",
        f"logger.tensorboard.save_dir={run_dir}",
        "experiment=it",
        "device=cpu",
        "compile.mode=null",  # Compiling on a CPU takes minutes; tests/unit/test_compile.py covers it.
        f"model.autoencoder.checkpoint_path={savi_checkpoint}",
        "model.max_steps=30",
        "model.num_seed=10",
        "model.update_freq=5",
        "model.num_updates=1",
        "model.eval_freq=20",
        "model.num_eval_episodes=1",
        "model.batch_size=2",
        "model.buffer_capacity=300",
        "model.num_context=2",
        "model.imagination_horizon=3",
        "model.env.max_episode_steps=8",
        *[item.replace("env.", "model.env.", 1) for item in FAKE_ENV],
    ]
    for component in ["dynamics_predictor", "actor", "critic", "reward_predictor"]:
        overrides += [f"model.{component}.{item}" for item in PREDICTOR]
    overrides += [f"{key}={value}" for key, value in values.items()]
    return overrides


@pytest.fixture(scope="module")
def savi_checkpoint(tmp_path_factory):
    """A toy SAVi checkpoint for 16x16 images and 2-D actions, saved like train_autoencoder does."""
    return save_savi_checkpoint(tmp_path_factory.mktemp("savi") / "latest.ckpt")


def save_savi_checkpoint(path, action_dim=2):
    import functools
    from lightning import Trainer

    savi = make_tiny_savi(action_dim=action_dim)
    module = train_autoencoder.AutoencoderModule(savi, functools.partial(torch.optim.Adam, lr=1e-3))
    trainer = Trainer(
        accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False, max_steps=0
    )
    trainer.strategy.connect(module)
    trainer.save_checkpoint(path)
    return path


# ---------------------------------------------------------------------------- generate_dataset
@pytest.mark.parametrize("save_format, num_workers", [("npz", 1), ("png", 2)])
def test_generate_dataset(monkeypatch, tmp_path, save_format, num_workers):
    out = tmp_path / "dataset"
    run_entry_point(
        monkeypatch,
        generate_dataset.generate_dataset,
        [
            f"hydra.run.dir={out}",
            f"save_format={save_format}",
            "num_train=3",
            "num_val=2",
            "num_test=1",
            f"num_workers={num_workers}",
            "env.max_episode_steps=4",
            *FAKE_ENV,
        ],
    )
    for split, count in [("train", 3), ("val", 2), ("test", 1)]:
        assert sorted(os.listdir(out / split)) == [str(i) for i in range(count)]
    files = os.listdir(out / "train" / "0")
    assert "episode.npz" in files and (("0.png" in files) == (save_format == "png"))


# --------------------------------------------------------------------------- train_autoencoder
AUTOENCODER = [
    "autoencoder.corrector.num_slots=2",
    "autoencoder.corrector.slot_dim=8",
    "autoencoder.corrector.hidden_dim=8",
    "autoencoder.encoder.num_channels=[4,4]",
    "autoencoder.encoder.kernel_sizes=[3,3]",
    "autoencoder.encoder.strides=[1,1]",
    "autoencoder.encoder.feature_dim=8",
    "autoencoder.decoder.num_channels=[4,4]",
    "autoencoder.decoder.kernel_sizes=[3,3]",
    "autoencoder.decoder.strides=[1,1]",
]


@pytest.mark.parametrize("wandb", [False, True])
def test_train_autoencoder(monkeypatch, tmp_path, wandb):
    from tests.conftest import write_image_dataset

    write_image_dataset(tmp_path / "data" / "run", "npz", num_episodes=(4, 2))
    fake_wandb = fakes.install_fake_wandb(monkeypatch)
    out = tmp_path / "out"
    run_entry_point(
        monkeypatch,
        train_autoencoder.train,
        [
            f"hydra.run.dir={out}",
            f"logger.tensorboard.save_dir={out}",
            f"logger.log_to_wandb={wandb}",
            f"dataset.path={tmp_path / 'data'}",
            "dataset.data_dir=run",
            "dataset.num_workers=0",
            "dataset.batch_size=2",
            "dataset.sequence_length=3",
            "trainer.max_epochs=1",
            "trainer.check_val_every_n_epoch=1",
            "trainer.accelerator=cpu",
            "callbacks.checkpoint.every_n_epochs=1",
            "callbacks.savi_decomposition.every_n_epochs=1",
            "model.scheduler.scheduler.warmup_epochs=1",
            *AUTOENCODER,
        ],
    )
    checkpoints = list(out.glob("logs/version_0/checkpoints/train_autoencoder-epoch=00-valid_loss=*.ckpt"))
    assert len(checkpoints) == 1
    autoencoder = train_autoencoder.load_autoencoder(str(checkpoints[0]))
    assert (autoencoder.num_slots, autoencoder.slot_dim) == (2, 8)
    assert list(autoencoder.decoder.image_size) == [16, 16]
    assert any(name.startswith("reconstruction-step=") for name in os.listdir(out / "logs/version_0/images"))
    assert [call[0] for call in fake_wandb.calls] == (["init", "finish"] if wandb else [])


# ------------------------------------------------------------------------- train_sold / evaluate
@pytest.fixture(scope="module")
def sold_run(tmp_path_factory, savi_checkpoint):
    monkeypatch = pytest.MonkeyPatch()
    run_dir = tmp_path_factory.mktemp("sold")
    try:
        run_entry_point(monkeypatch, train_sold.train, sold_overrides(run_dir, savi_checkpoint))
    finally:
        monkeypatch.undo()
    return run_dir


def test_train_sold_writes_checkpoints_and_metrics(sold_run):
    checkpoints_dir = sold_run / "logs" / "version_0" / "checkpoints"
    names = sorted(p.name for p in checkpoints_dir.iterdir())
    # keep_last_checkpoint_only defaults to True: the step-0 checkpoint was deleted.
    assert [n.split("-eval")[0] for n in names if n.endswith(".ckpt")] == ["sold-steps=24-episode=3"]
    assert not any(name.startswith("replay_buffer-") for name in names)  # save_replay_buffer defaults to False.
    checkpoint = torch.load(sorted(checkpoints_dir.glob("sold-steps=24*.ckpt"))[0], weights_only=False)
    assert checkpoint["replay_buffer_path"] == (sold_run / "logs" / "version_0" / "replay_buffer").resolve()
    assert (checkpoint["replay_buffer_path"] / "bookkeeping.pt").exists()
    metrics = read_metrics(sold_run / "logs/version_0")
    assert any("eval/episode_return" in record for record in metrics)
    assert (sold_run / ".hydra" / "config.yaml").exists()


def test_train_sold_with_parallel_environments(monkeypatch, tmp_path, savi_checkpoint):
    run_entry_point(monkeypatch, train_sold.train, sold_overrides(tmp_path, savi_checkpoint, num_envs=2))
    records = [json.loads(line) for line in open(next(tmp_path.glob("logs/version_0/metrics/metrics.jsonl")))]
    assert max(record["step"] for record in records) >= 30 and any("eval/episode_return" in r for r in records)
    assert next(tmp_path.glob("logs/version_0/checkpoints/*.ckpt"))


def test_train_sold_with_bfloat16_autocast(monkeypatch, tmp_path, savi_checkpoint):
    calls = []
    autocast_stages = train_sold.SOLDModule.autocast_stages

    def recording_autocast_stages(self, dtype, stages):
        calls.append((dtype, list(stages)))
        autocast_stages(self, dtype, stages)

    monkeypatch.setattr(train_sold.SOLDModule, "autocast_stages", recording_autocast_stages)
    run_entry_point(
        monkeypatch, train_sold.train, sold_overrides(tmp_path, savi_checkpoint, **{"autocast.dtype": "bfloat16"})
    )
    assert calls == [("bfloat16", ["dynamics_loss", "reward_loss", "imagination", "actor_loss", "critic_loss", "act"])]
    records = read_metrics(tmp_path / "logs/version_0")
    losses = [record["train/actor_loss"] for record in records if "train/actor_loss" in record]
    assert losses and all(math.isfinite(loss) for loss in losses)


def test_train_sold_with_wandb(monkeypatch, tmp_path, savi_checkpoint):
    fake_wandb = fakes.install_fake_wandb(monkeypatch)
    precisions = []
    monkeypatch.setattr(torch, "set_float32_matmul_precision", precisions.append)
    run_entry_point(
        monkeypatch,
        train_sold.train,
        sold_overrides(tmp_path, savi_checkpoint, **{"logger.log_to_wandb": True, "model.max_steps": 10}),
    )
    assert [call[0] for call in fake_wandb.calls] == ["init", "finish"]
    assert precisions[0] == "high"  # TF32 matmuls.


def _checkpoint_without_replay_buffer(sold_run, tmp_path):
    """Copy of the newest checkpoint whose replay buffer folder is gone (bugs #4 and #6 make the other case fail)."""
    checkpoints_dir = sold_run / "logs" / "version_0" / "checkpoints"
    source = sorted(checkpoints_dir.glob("sold-steps=24*.ckpt"))[0]
    checkpoint = torch.load(source, weights_only=False)
    checkpoint["replay_buffer_path"] = tmp_path / "deleted_replay_buffer"
    target = tmp_path / source.name
    torch.save(checkpoint, target)
    return target


def test_train_sold_resumes_from_checkpoint(monkeypatch, tmp_path, sold_run, savi_checkpoint):
    checkpoint = _checkpoint_without_replay_buffer(sold_run, tmp_path)
    run_dir = tmp_path / "resumed"
    run_entry_point(
        monkeypatch,
        train_sold.train,
        sold_overrides(run_dir, savi_checkpoint, checkpoint=f"'{checkpoint}'", **{"model.max_steps": 45}),
    )
    names = [p.name.split("-eval")[0] for p in (run_dir / "logs/version_0/checkpoints").glob("*.ckpt")]
    # Continues counting at step 24 / episode 3 (with an empty replay buffer) and evaluates when step 40 is reached.
    assert names == ["sold-steps=40-episode=5"]


def test_evaluate_sold(monkeypatch, tmp_path, sold_run):
    checkpoint = _checkpoint_without_replay_buffer(sold_run, tmp_path)
    out = tmp_path / "eval"
    monkeypatch.chdir(tmp_path)
    run_entry_point(
        monkeypatch,
        evaluate_sold.evaluate,
        [
            f"hydra.run.dir={out}",
            f"checkpoint_path='{checkpoint}'",
            "eval_episodes=2",
            "env.max_episode_steps=8",
            *FAKE_ENV,
        ],
    )
    record = json.loads(open(out / "metrics.jsonl").read())
    assert record["step"] == 24 and record["checkpoint"] == str(checkpoint)
    assert len(record["episode_returns"]) == 2 and 0.0 <= record["success_rate"] <= 1.0
    assert {"timing/eval/action_selection_ms", "timing/eval/env_step_ms"} <= record.keys()
    videos = sorted(os.listdir(out / "videos" / checkpoint.stem))
    assert videos == ["episode_high_res_0.mp4", "episode_high_res_1.mp4", "episode_obs_0.mp4", "episode_obs_1.mp4"]


# ----------------------------------------------------------------------------------- instructions
INSTRUCTIONS = ["+model.env.language={memory_length:8}", "model.language={embed_dim:4}"]


@pytest.mark.usefixtures("fake_language")
def test_train_and_evaluate_sold_with_instructions(monkeypatch, tmp_path, savi_checkpoint):
    from sold.data.ring_buffer import RingBufferDataset

    run_entry_point(monkeypatch, train_sold.train, sold_overrides(tmp_path, savi_checkpoint) + INSTRUCTIONS)
    checkpoint = next(tmp_path.glob("logs/version_0/checkpoints/*.ckpt"))
    assert torch.load(checkpoint, weights_only=False)["hyper_parameters"]["language"] == {"embed_dim": 4}
    fields = RingBufferDataset.load_bookkeeping(tmp_path / "logs/version_0/replay_buffer")["info"].fields
    assert set(fields) == {"obs", "tokens", "action", "reward"}
    out = tmp_path / "eval"
    monkeypatch.chdir(tmp_path)
    evaluation = [f"hydra.run.dir={out}", f"checkpoint_path='{checkpoint}'", "eval_episodes=2"]
    evaluation += ["env.max_episode_steps=8", *FAKE_ENV, "+env.language={memory_length:8}"]
    run_entry_point(monkeypatch, evaluate_sold.evaluate, evaluation)
    assert len(json.loads(open(out / "metrics.jsonl").read())["episode_returns"]) == 2
    assert len(os.listdir(out / "videos" / checkpoint.stem)) == 4


@pytest.mark.mujoco
@pytest.mark.parametrize("reveal", ["stream", "full"])
def test_train_sold_on_reach_multicolor(monkeypatch, tmp_path, reveal):
    pytest.importorskip("multi_object_fetch.language")
    savi = save_savi_checkpoint(tmp_path / "savi.ckpt", action_dim=4)  # Fetch actions: x, y, z, gripper.
    fake_env = tuple(item.split("=")[0].replace("env.", "model.env.", 1) for item in FAKE_ENV)
    overrides = [item for item in sold_overrides(tmp_path, savi) if not item.startswith(fake_env)]
    overrides += [
        "environment@model.env=reach_multicolor",
        "model.env.image_size=[16,16]",
        f"model.env.language.reveal={reveal}",
        "model.language={embed_dim:4}",
    ]
    run_entry_point(monkeypatch, train_sold.train, overrides)
    records = read_metrics(tmp_path / "logs/version_0")
    assert any("eval/episode_return" in record for record in records)
    losses = [record["train/reward_loss"] for record in records if "train/reward_loss" in record]
    assert losses and all(math.isfinite(loss) for loss in losses)


def test_evaluate_sold_directory_without_success_info(monkeypatch, tmp_path, sold_run):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    latest = torch.load(_checkpoint_without_replay_buffer(sold_run, checkpoints), weights_only=False)
    # A second checkpoint of the directory (the run itself kept only its last one).
    torch.save({**latest, "num_steps": 0}, checkpoints / "sold-steps=0-episode=0-eval_episode_return=0.0.ckpt")
    original_step = fakes.FakePixelEnv.step
    monkeypatch.setattr(fakes.FakePixelEnv, "step", lambda self, action: original_step(self, action)[:3] + ({},))
    out = tmp_path / "eval"
    monkeypatch.chdir(tmp_path)
    run_entry_point(
        monkeypatch,
        evaluate_sold.evaluate,
        [
            f"hydra.run.dir={out}",
            f"checkpoint_path={checkpoints}",
            "eval_episodes=1",
            "env.max_episode_steps=8",
            *FAKE_ENV,
        ],
    )
    records = [json.loads(line) for line in open(out / "metrics.jsonl")]
    assert sorted(r["step"] for r in records) == [0, 24]
    assert all("success_rate" not in r for r in records)
