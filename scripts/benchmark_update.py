"""Time SOLD's training update on a synthetic batch, eagerly or with compiled training stages.

    python scripts/benchmark_update.py [--compile-mode MODE] [--compile-stages a,b] [--autocast bfloat16]
                                       [--no-fused] [--warmup N] [--updates N] [HYDRA_OVERRIDES...]

The model comes from configs/train_sold.yaml with preset=push_red (the PushRed runs' 4-layer setup) and the given
overrides, with a stub environment of the run's action space: no environment, replay buffer or logger is needed.
Prints the compile time, the milliseconds per update, the per-stage timings, torch.compile's counters and the losses of
one more update.
Run one mode per process (compiled code is cached per process).
"""

import argparse
import gym
from hydra import compose, initialize_config_dir
import hydra.utils
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import time
import torch

REPO_DIR = Path(__file__).resolve().parents[1]


class StubEnv(gym.Env):
    """Only what SOLDModule reads from an environment: the action space and the episode length."""

    def __init__(self, action_dim: int, max_episode_steps: int) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32)
        self.max_episode_steps = max_episode_steps


def synthetic_batch(sold, image_size, action_dim, device):
    shape = (sold.batch_size, sold.sequence_length)
    actions = torch.rand(*shape, action_dim, device=device) * 2 - 1
    rewards = torch.rand(*shape, device=device)
    actions[:, 0], rewards[:, 0] = float("nan"), float("nan")
    obs = torch.randint(0, 256, (*shape, 3, *image_size), dtype=torch.uint8, device=device)
    return {"obs": obs, "action": actions, "reward": rewards}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compile-mode", default=None, help='e.g. "default" or "reduce-overhead"; eager if unset')
    parser.add_argument("--compile-stages", default=None, help="comma-separated stages; all if unset")
    parser.add_argument("--autocast", default=None, help='"bfloat16": autocast of the config\'s autocast.stages')
    parser.add_argument("--no-fused", action="store_true", help="unfused Adam")
    parser.add_argument("--warmup", type=int, default=10, help="untimed updates first (they compile)")
    parser.add_argument("--updates", type=int, default=50, help="timed updates")
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of configs/train_sold.yaml")
    args = parser.parse_args()

    savi = REPO_DIR / "checkpoints/autoencoder/savi/push_red/latest.ckpt"
    with initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base=None):
        overrides = ["preset=push_red", f"model.autoencoder.checkpoint_path={savi}", *args.overrides]
        cfg = compose("train_sold", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    env_cfg = cfg.model.pop("env")
    image_size = tuple(env_cfg.image_size)

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sold = hydra.utils.instantiate(cfg.model, env=StubEnv(args.action_dim, env_cfg.max_episode_steps)).to(device)
    sold.train()
    sold.fused_optimizers = device.type == "cuda" and not args.no_fused
    sold.optimizers = sold.configure_optimizers()
    stages = args.compile_stages.split(",") if args.compile_stages else None
    if args.compile_mode is not None:
        sold.compile_stages(args.compile_mode, stages)
    if args.autocast is not None:  # After compiling, as in fit.
        sold.autocast_stages(args.autocast, cfg.autocast.stages)
    batch = synthetic_batch(sold, image_size, args.action_dim, device)

    def run(num_updates: int) -> float:
        """Seconds per update."""
        start = time.perf_counter()
        for _ in range(num_updates):
            if sold.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            sold.training_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - start) / num_updates

    first = run(1)
    run(args.warmup - 1)
    sold.timings.summary()  # Drop the warm-up timings.
    per_update = run(args.updates)
    stage_ms = sold.timings.summary()

    from torch._dynamo.utils import counters

    compiled = (stages or "all") if args.compile_mode else "-"
    print(f"mode={args.compile_mode} stages={compiled} autocast={args.autocast} fused={sold.fused_optimizers}")
    print(f"first update {first:.1f} s (compiles), then {per_update * 1000:.1f} ms per update")
    for name, ms in sorted(stage_ms.items()):
        print(f"  {name:28s} {ms:8.1f} ms")
    print(f"graph breaks: {sum(counters['graph_break'].values())}, unique graphs: {counters['stats']['unique_graphs']}")
    print(f"cudagraph skips: {dict(counters['inductor']).get('cudagraph_skips', 0)}")
    if device.type == "cuda":
        print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    if sold.cudagraphs:
        torch.compiler.cudagraph_mark_step_begin()
    torch.manual_seed(0)  # The same random numbers in every mode (up to compiled code's own).
    losses = {key: value.item() for key, value in sold.training_step(batch).items() if key.endswith("_loss")}
    print("losses of one more update: " + ", ".join(f"{key} {value:.5g}" for key, value in sorted(losses.items())))


if __name__ == "__main__":
    main()
