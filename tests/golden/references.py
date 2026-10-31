"""Golden references: outputs of the released checkpoints and of SOLD's losses on fixed inputs.

The tests in test_golden.py recompute these and compare them with the tensors stored in tests/golden/data/.
A refactor that renames modules, changes state-dict keys or alters the math makes them fail.

If an output change is intended, regenerate the references from the repository root and review the diff:

    PYTHONPATH=sold:. python -m tests.golden.references

LEGACY_CHECKPOINT is a tiny SOLD checkpoint in the checkpoint format from before the refactoring (commit 4f6b757),
with fine-tuning on (five optimizers). Lightning checkpoints pickle SOLDModule's constructor arguments and module paths,
so it fails to load when a refactor changes either without a compatibility shim. It was generated once with

    PYTHONPATH=sold:. python -m tests.golden.references --legacy-checkpoint

Never regenerate it: a checkpoint written by newer code would no longer test old checkpoints. Its reference outputs
are regenerated with the others.
"""

from pathlib import Path
import sys
import tempfile

import torch

DATA_DIR = Path(__file__).parent / "data"
REPO_DIR = Path(__file__).resolve().parents[2]
SAVI_CHECKPOINTS = {"savi_push_red": "autoencoder/savi/push_red/latest.ckpt",
                    "savi_reach_red": "autoencoder/savi/reach_red/latest.ckpt"}
SOLD_CHECKPOINT = "sold/reach_red/latest.ckpt"
LEGACY_CHECKPOINT = DATA_DIR / "legacy_tiny_sold.ckpt"


def _images(seed, batch=1, length=3, size=64):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, length, 3, size, size, generator=generator)


@torch.no_grad()
def savi_outputs(checkpoint_path):
    from train_autoencoder import AutoencoderModule
    # Not load_autoencoder(): it restores onto the GPU the checkpoint was saved from (see bug #27).
    savi = AutoencoderModule.load_from_checkpoint(str(checkpoint_path), map_location="cpu").autoencoder.eval()
    images = _images(0)
    actions = torch.zeros(1, 2, savi.predictor.action_dim)
    slots = savi.encode(images, actions)
    decoded = savi.decode(slots)
    return {"slots": slots, "reconstructions": decoded["reconstructions"], "masks": decoded["masks"]}


def _sold_env(image_size, action_dim, max_episode_steps):
    from tests.fakes import make_fake_env
    env = make_fake_env(image_size=image_size, max_episode_steps=max_episode_steps)
    env.unwrapped.action_space = env.action_space = type(env.action_space)(-1.0, 1.0, (action_dim,),
                                                                           env.action_space.dtype)
    return env


def load_sold(checkpoint_path, image_size=(64, 64), action_dim=4, max_episode_steps=50, **overrides):
    """Load a SOLD checkpoint for a fake environment matching its image size and action space."""
    from train_sold import SOLDModule
    env = _sold_env(image_size, action_dim, max_episode_steps)
    return SOLDModule.load_from_checkpoint(str(checkpoint_path), env=env, map_location="cpu", **overrides)


def load_legacy_sold(**overrides):
    return load_sold(LEGACY_CHECKPOINT, image_size=(16, 16), action_dim=2, max_episode_steps=8, **overrides)


@torch.no_grad()
def _sold_model_outputs(sold, images, num_context):
    """Outputs of SOLD's actor, critic, reward predictor and dynamics model for fixed images."""
    sold = sold.eval()
    batch_size, sequence_length = images.shape[:2]
    action_dim = sold.actor.lower_bound.numel()
    slots = sold.autoencoder.encode(images, torch.zeros(batch_size, sequence_length - 1, action_dim))
    actions = torch.linspace(-1, 1, (sequence_length - 1) * action_dim).view(1, sequence_length - 1, action_dim)
    actor = sold.actor(slots)
    return {"actor_mean": actor.mean,
            "actor_stddev": actor.base_dist.stddev,
            "critic_mean": sold.critic(slots).mean,
            "reward_mean": sold.reward_predictor(slots).mean,
            "predicted_slots": sold.dynamics_predictor.predict_slots(slots, actions, steps=2, num_context=num_context)}


def sold_outputs(checkpoint_path):
    return _sold_model_outputs(load_sold(checkpoint_path), _images(1, length=5), num_context=3)


def legacy_sold_outputs():
    return _sold_model_outputs(load_legacy_sold(), _images(3, length=4, size=16), num_context=2)


def tiny_sold_losses():
    """SOLD's losses for the tiny test model on a fixed batch (no optimizer steps)."""
    from tests.conftest import make_tiny_sold
    torch.manual_seed(0)
    sold = make_tiny_sold()
    sold.after_eval = False
    generator = torch.Generator().manual_seed(2)
    images = torch.rand(2, sold.sequence_length, 3, 16, 16, generator=generator)
    actions = torch.rand(2, sold.sequence_length, 2, generator=generator) * 2 - 1
    rewards = torch.rand(2, sold.sequence_length, generator=generator)
    actions[:, 0], rewards[:, 0] = float("nan"), float("nan")
    torch.manual_seed(1)
    outputs = sold.autoencoder(images, actions[:, 1:])
    slots = outputs["slots"].detach()
    losses = {"reconstruction_loss": torch.mean((outputs["reconstructions"] - images) ** 2)}
    losses |= sold.compute_dynamics_loss(images, slots, actions)
    losses |= sold.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
    imagination = sold.imagine_ahead(slots, actions)
    losses |= sold.compute_actor_loss(imagination.lambda_returns, imagination.predicted_values_targ,
                                      imagination.action_log_probs, imagination.action_entropies)
    losses |= sold.compute_critic_loss(imagination.predicted_values_dist, imagination.lambda_returns,
                                       imagination.predicted_values_targ)
    return {key: torch.as_tensor(value).detach() for key, value in losses.items()}


def tiny_sold_training():
    """Parameters of the tiny SOLD after 20 environment steps of training with fine-tuning (five optimizers)."""
    from tests.conftest import make_online_trainer, make_tiny_sold
    from tests.fakes import make_fake_env
    torch.manual_seed(0)
    env = make_fake_env()
    env.action_space.seed(0)  # Seed-phase actions come from gym's own generator.
    sold = make_tiny_sold(env, finetune_autoencoder=True, max_steps=20)
    with tempfile.TemporaryDirectory() as tmp_dir:
        make_online_trainer(Path(tmp_dir)).fit(sold)
    names = ["autoencoder", "dynamics_predictor", "reward_predictor", "actor", "critic", "critic_target"]
    outputs = {name: torch.cat([p.detach().flatten() for p in getattr(sold, name).parameters()]) for name in names}
    outputs["return_moments"] = torch.stack([sold.return_moments.low, sold.return_moments.high])
    return outputs


def compute_all():
    references = {name: savi_outputs(REPO_DIR / "checkpoints" / path) for name, path in SAVI_CHECKPOINTS.items()}
    references["sold_reach_red"] = sold_outputs(REPO_DIR / "checkpoints" / SOLD_CHECKPOINT)
    references["legacy_tiny_sold"] = legacy_sold_outputs()
    references["tiny_sold_losses"] = tiny_sold_losses()
    references["tiny_sold_training"] = tiny_sold_training()
    return references


def generate_legacy_checkpoint():
    """Train the tiny SOLD for a few steps with fine-tuning and save it (see the module docstring)."""
    from tests.conftest import make_online_trainer, make_tiny_sold
    torch.manual_seed(0)
    sold = make_tiny_sold(finetune_autoencoder=True, max_steps=20)
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = make_online_trainer(Path(tmp_dir))
        trainer.fit(sold)
        trainer.save_checkpoint(LEGACY_CHECKPOINT)
    print(f"wrote {LEGACY_CHECKPOINT}")


if __name__ == "__main__":
    if "--legacy-checkpoint" in sys.argv:
        generate_legacy_checkpoint()
    DATA_DIR.mkdir(exist_ok=True)
    for name, tensors in compute_all().items():
        torch.save(tensors, DATA_DIR / f"{name}.pt")
        print(f"wrote {DATA_DIR / name}.pt: {', '.join(f'{k}{tuple(v.shape)}' for k, v in tensors.items())}")
