"""Drift test: compare compiled SOLD training updates with eager ones on PushRed data (sold.utils.drift).

    python scripts/drift_test.py collect --out DIR [--checkpoint CKPT] [--episodes 200] [HYDRA_OVERRIDES...]
    python scripts/drift_test.py run --data DIR/data.npz --variant VARIANT --out FILE [--checkpoint CKPT]
                                     [--updates 1000] [HYDRA_OVERRIDES...]

collect plays PushRed episodes (preset=push_red) with random actions, or with the checkpoint's policy, and saves them.
run trains models from the same start (the seeded initial model, or the checkpoint with its optimizer states) on the
same batches of that data, one batch per update (batch k drawn with seed k), and compares them update by update:

    resync              E (eager), E2 (eager again: the floor) and G (CUDA graphs drawing eager random numbers),
                        re-synced to E before every update: every stage compares exactly.
    resync-production   E and P (CUDA graphs as in training, with Inductor's random numbers), re-synced: the stages
                        without random numbers compare exactly, the others are checked for a bias.
    free                E, E_seed (eager, other random numbers) and P, training freely: how far P drifts from E,
                        against how far E_seed does, and scalar biases.

Writes one JSON record per update to FILE and a summary to FILE with the suffix .summary.json; prints the summary.
"""

import argparse
from benchmark_update import REPO_DIR, StubEnv
from hydra import compose, initialize_config_dir
import hydra.utils
import json
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from sold.data.ring_buffer import RingBufferDataset
from sold.train_sold import SOLDModule
from sold.utils import drift
import sys
import tempfile
import time
import torch

VARIANTS = {  # Models (the first is the reference) and whether they are re-synced to it before every update.
    "resync": (("E", "E2", "G"), True),
    "resync-production": (("E", "P"), True),
    "free": (("E", "E_seed", "P"), False),
}
COMPILED = {"G", "P"}
SEED_OFFSETS = {"E_seed": 1_000_000}
# Stages that draw no random numbers, so that P matches E in them under re-sync.
DETERMINISTIC = ("stage/autoencoder/", "stage/reward_loss/", "grad/reward_predictor", "update/reward_predictor")
MIN_FREE_GPU_MEMORY_GB = 5


def load_config(overrides: list[str]):
    """The resolved run config (preset=push_red with overrides) and its environment config, split from the model's."""
    savi = REPO_DIR / "checkpoints/autoencoder/savi/push_red/latest.ckpt"
    with initialize_config_dir(config_dir=str(REPO_DIR / "configs"), version_base=None):
        overrides = ["preset=push_red", f"model.autoencoder.checkpoint_path={savi}", *overrides]
        cfg = compose("train_sold", overrides=overrides)
    OmegaConf.set_struct(cfg, False)
    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.env, resolve=True))  # Resolves its seed.
    del cfg.model.env
    return cfg, env_cfg


def load_model(cfg, checkpoint: str | None, env, device: torch.device) -> SOLDModule:
    """The checkpoint's model with its optimizer states, or the seeded initial model of cfg."""
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)  # Loaded per model: no shared modules.
        model = SOLDModule(env=env, **saved["hyper_parameters"]).to(device)
    else:
        torch.manual_seed(cfg.seed)
        model = hydra.utils.instantiate(cfg.model, env=env).to(device)
    model.train()
    model.fused_optimizers = device.type == "cuda"
    model.optimizers = model.configure_optimizers()
    if checkpoint is not None:
        model.restore(saved)
    return model


@torch.no_grad()
def collect(args, cfg, env_cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = hydra.utils.instantiate(env_cfg)
    env.action_space.seed(cfg.seed)
    policy = load_model(cfg, args.checkpoint, env, device) if args.checkpoint else None
    episodes: dict[str, list[np.ndarray]] = {"obs": [], "action": [], "reward": []}
    for index in range(args.episodes):
        obs, done, first = env.reset(), False, True
        nan_action = np.full(env.action_space.shape, np.nan, np.float32)
        episode = {"obs": [obs.numpy()], "action": [nan_action], "reward": [np.float32(np.nan)]}
        while not done:
            if policy is None:
                action = torch.from_numpy(env.action_space.sample().astype(np.float32))
            else:
                action = policy._policy(obs, is_first=first, mode="train").cpu()
            obs, reward, done, _ = env.step(action)
            first = False
            episode["obs"].append(obs.numpy())
            episode["action"].append(action.numpy())
            episode["reward"].append(np.float32(reward))
        for key, values in episode.items():
            episodes[key].append(np.stack(values))
        print(f"episode {index + 1}/{args.episodes}: return {np.nansum(episode['reward']):.2f}", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(args.out) / "data.npz", **{key: np.stack(values) for key, values in episodes.items()})


def batches(data: dict[str, np.ndarray], model: SOLDModule, num_updates: int, device: torch.device):
    """Batch k is drawn with a generator seeded with k, so that every process trains on the same batches."""
    num_episodes, length = data["obs"].shape[:2]
    with tempfile.TemporaryDirectory() as save_path:
        buffer = RingBufferDataset(num_episodes * length, model.batch_size, model.sequence_length, save_path=save_path)
        for index in range(num_episodes):
            buffer.add_episode({key: list(torch.from_numpy(values[index])) for key, values in data.items()})
        for step in range(num_updates):
            batch = buffer.sample(torch.Generator().manual_seed(step))
            yield {key: value.to(device) for key, value in batch.items()}


def summarize(records: list[dict[str, float]], variant: str) -> dict:
    if variant == "resync":
        return {"G": drift.summarize(records, "G", floor_model="E2"), "E2": drift.summarize(records, "E2")}
    scalar_keys = sorted(key[2:] for key in records[0] if key.startswith("E/"))
    if variant == "resync-production":
        exact = {k: v for k, v in drift.summarize(records, "P").items() if k.startswith(DETERMINISTIC)}
        return {"P_exact": exact, "P_bias": {key: drift.bias(records, key, "P", "E") for key in scalar_keys}}
    distances = {}
    for model in ("P", "E_seed"):
        prefix = f"{model}/err/"
        keys = [
            k for k in records[0] if k.startswith(f"{prefix}state/") or (k.startswith(prefix) and k.endswith("_loss"))
        ]
        marks = sorted({min(n, len(records)) - 1 for n in (1, 10, 100, 500, 1000, 2000, len(records))})
        distances[model] = {key: {int(records[m]["step"]): records[m][key] for m in marks} for key in keys}
    bias = {model: {key: drift.bias(records, key, model, "E") for key in scalar_keys} for model in ("P", "E_seed")}
    return {"distances": distances, "bias": bias}


def _bias_line(key: str, value: dict[str, float]) -> str:
    return f"   {key:60s} mean {value['mean']:+.3e}  se {value['standard_error']:.1e}  z {value['z']:+.1f}"


def print_summary(summary: dict, variant: str) -> None:
    if variant in ("resync", "resync-production"):
        for name, metrics in summary.items():
            if name.endswith("_bias"):
                large = {k: v for k, v in metrics.items() if abs(v["z"]) > 3}
                print(f"== {name}: {len(large)} of {len(metrics)} scalars with |z| > 3")
                for key, value in sorted(large.items(), key=lambda item: -abs(item[1]["z"])):
                    print(_bias_line(key, value))
                continue
            flagged = [k for k, v in metrics.items() if v["flagged"]]
            print(f"== {name}: {len(flagged)} of {len(metrics)} metrics flagged {flagged}")
            print(f"   {'metric':60s} {'median':>9s} {'p99':>9s} {'max':>9s} {'step':>5s} {'floor':>9s}")
            for key, value in sorted(metrics.items(), key=lambda item: -item[1]["max"])[:30]:
                print(
                    f"   {key:60s} {value['median']:9.1e} {value['p99']:9.1e} {value['max']:9.1e} "
                    f"{value['step_of_max']:5d} {value['floor']:9.1e}{'  FLAGGED' if value['flagged'] else ''}"
                )
        return
    for model, metrics in summary["distances"].items():
        print(f"== {model}: distance to E at steps")
        for key, values in metrics.items():
            print(f"   {key:60s} " + " ".join(f"{step}:{value:.2e}" for step, value in values.items()))
    for model, metrics in summary["bias"].items():
        large = {k: v for k, v in metrics.items() if abs(v["z"]) > 3}
        print(f"== {model} bias against E: {len(large)} of {len(metrics)} scalars with |z| > 3")
        for key, value in sorted(large.items(), key=lambda item: -abs(item[1]["z"]))[:30]:
            print(_bias_line(key, value))


def run(args, cfg, env_cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    names, resync = VARIANTS[args.variant]
    # G draws eager random numbers inside its CUDA graphs; P uses Inductor's, as in training.
    torch._inductor.config.fallback_random = args.variant == "resync"
    data = dict(np.load(args.data))
    env = StubEnv(data["action"].shape[-1], env_cfg.max_episode_steps)
    models = {name: load_model(cfg, args.checkpoint, env, device) for name in names}
    for name in COMPILED & set(names):
        models[name].compile_stages("reduce-overhead")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records, start = [], time.perf_counter()
    stream = drift.lockstep(models, batches(data, models["E"], args.updates, device), resync, SEED_OFFSETS)
    with open(out, "w") as file:
        for record in stream:
            records.append(record)
            file.write(json.dumps(record) + "\n")
            step = int(record["step"])
            if device.type == "cuda" and step == 2:  # Compiled and recorded by now.
                free_gb = torch.cuda.mem_get_info()[0] / 2**30
                print(f"free GPU memory {free_gb:.1f} GB", flush=True)
                if free_gb < MIN_FREE_GPU_MEMORY_GB:
                    sys.exit(f"Stopped: less than {MIN_FREE_GPU_MEMORY_GB} GB of GPU memory free.")
            if (step + 1) % 100 == 0:
                print(f"update {step + 1}/{args.updates}, {time.perf_counter() - start:.0f} s", flush=True)
    summary = summarize(records, args.variant)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print_summary(summary, args.variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["collect", "run"])
    parser.add_argument("--out", required=True, help="collect: a directory; run: the records' JSONL file")
    parser.add_argument("--checkpoint", default=None, help="start from this SOLD checkpoint (its policy collects)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--data", help="run: the data.npz written by collect")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="resync")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("overrides", nargs="*", help="Hydra overrides of configs/train_sold.yaml")
    args = parser.parse_args()
    (collect if args.command == "collect" else run)(args, *load_config(args.overrides))


if __name__ == "__main__":
    main()
