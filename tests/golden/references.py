"""Golden references: outputs of the released checkpoints and of SOLD's losses on fixed inputs.

The tests in test_golden.py recompute these and compare them with the tensors stored in tests/golden/data/.
A refactor that renames modules, changes state-dict keys or alters the math makes them fail.

If an output change is intended, regenerate the references from the repository root and review the diff:

    PYTHONPATH=sold:. python -m tests.golden.references
"""

from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent / "data"
REPO_DIR = Path(__file__).resolve().parents[2]
SAVI_CHECKPOINTS = {"savi_push_red": "autoencoder/savi/push_red/latest.ckpt",
                    "savi_reach_red": "autoencoder/savi/reach_red/latest.ckpt"}
SOLD_CHECKPOINT = "sold/reach_red/latest.ckpt"


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


def load_sold(checkpoint_path):
    from tests.fakes import make_fake_env
    from train_sold import SOLDModule
    env = make_fake_env(image_size=(64, 64), max_episode_steps=50)
    env.unwrapped.action_space = env.action_space = type(env.action_space)(-1.0, 1.0, (4,), env.action_space.dtype)
    return SOLDModule.load_from_checkpoint(str(checkpoint_path), env=env, map_location="cpu").eval()


@torch.no_grad()
def sold_outputs(checkpoint_path):
    sold = load_sold(checkpoint_path)
    images = _images(1, length=5)
    slots = sold.autoencoder.encode(images, torch.zeros(1, 4, 4))
    actions = torch.linspace(-1, 1, 16).view(1, 4, 4)
    return {"actor_mean": sold.actor(slots).mean,
            "actor_stddev": sold.actor(slots).base_dist.stddev,
            "critic_mean": sold.critic(slots).mean,
            "reward_mean": sold.reward_predictor(slots).mean,
            "predicted_slots": sold.dynamics_predictor.predict_slots(slots, actions, steps=2, num_context=3)}


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
    returns, values_targ, values_dist, log_probs, entropies = sold.imagine_ahead(slots, actions)
    losses |= sold.compute_actor_loss(returns, values_targ, log_probs, entropies)
    losses |= sold.compute_critic_loss(values_dist, returns, values_targ)
    return {key: torch.as_tensor(value).detach() for key, value in losses.items()}


def compute_all():
    references = {name: savi_outputs(REPO_DIR / "checkpoints" / path) for name, path in SAVI_CHECKPOINTS.items()}
    references["sold_reach_red"] = sold_outputs(REPO_DIR / "checkpoints" / SOLD_CHECKPOINT)
    references["tiny_sold_losses"] = tiny_sold_losses()
    return references


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, tensors in compute_all().items():
        torch.save(tensors, DATA_DIR / f"{name}.pt")
        print(f"wrote {DATA_DIR / name}.pt: {', '.join(f'{k}{tuple(v.shape)}' for k, v in tensors.items())}")
