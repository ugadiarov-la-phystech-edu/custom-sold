"""Rebuild a RingBufferDataset's bookkeeping (head, tail, episode boundaries, info) from its on-disk memmaps.

    python -m scripts.rebuild_replay_buffer_bookkeeping BUFFER_DIR CAPACITY [--output FILE] [--check]
                                                        [--episode-length N] [--sample-episodes N] [--seed S]

BUFFER_DIR holds obs.dat (uint8, [capacity, 3, side, side]), action.dat (float32, [capacity, action_dim]) and
reward.dat (float32, [capacity]) as written by sold.data.ring_buffer.RingBufferDataset; the shapes are inferred from
the file sizes. Episodes start at the rows with a NaN action (reset entries) and run to the next start, circularly.
When the ring has wrapped and the capacity is not a multiple of the episode length, one gap also holds the seam: the
partially overwritten old data between the newest episode's end (tail) and the oldest surviving start (head). It is
the one gap whose length differs from the others, cross-checked against the frame content (consecutive frames of an
episode are similar, the frames on either side of the seam are not). In a ring that has not wrapped, the seam is the
memmap's zero fill after the newest episode.
Saves {"capacity", "info", "head", "tail", "episode_boundaries"} with torch.save: the arguments of
RingBufferDataset.load_from_disk_storage, with boundaries (start, start + length) as store_episode records them.
--check validates the result against the data and the saved file, and exits with status 1 on a failure.
"""

import argparse
from collections import Counter, deque
import math
import numpy as np
import os
from sold.data.info import EpisodeDatasetInfo, Fields
import sys
import torch

KEYS = ("obs", "action", "reward")
DTYPES = {"obs": np.uint8, "action": np.float32, "reward": np.float32}


def open_memmaps(buffer_dir: str, capacity: int) -> dict[str, np.memmap]:
    """Read-only memmaps of the three fields, with shapes inferred from the file sizes."""
    sizes = {key: os.path.getsize(os.path.join(buffer_dir, f"{key}.dat")) for key in KEYS}
    for key, size in sizes.items():
        if size == 0 or size % capacity:
            raise ValueError(f"{key}.dat has {size} bytes, not a positive multiple of the capacity {capacity}.")
    side = math.isqrt(sizes["obs"] // capacity // 3)
    if side * side * 3 != sizes["obs"] // capacity:
        raise ValueError(f"obs.dat rows have {sizes['obs'] // capacity} bytes, not a square color image.")
    if sizes["action"] % 4 or sizes["reward"] != 4 * capacity:
        raise ValueError("action.dat/reward.dat sizes do not match float32 rows.")
    shapes = {
        "obs": (capacity, 3, side, side),  # Channels first, as the Pixels wrapper stores frames.
        "action": (capacity, sizes["action"] // capacity // 4),
        "reward": (capacity,),
    }
    return {
        key: np.memmap(os.path.join(buffer_dir, f"{key}.dat"), dtype=DTYPES[key], mode="r", shape=shapes[key])
        for key in KEYS
    }


def find_episodes(action: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Episode starts (rows with a NaN action), the circular gap after each, the seam gap's index, the common length."""
    nan_rows = np.isnan(action).any(axis=1)
    if not np.array_equal(nan_rows, np.isnan(action).all(axis=1)):
        raise ValueError("Found rows whose action is only partly NaN.")
    starts = np.flatnonzero(nan_rows)
    if len(starts) < 2:
        raise ValueError(f"Found {len(starts)} episode starts; need at least two.")
    gaps = np.diff(np.append(starts, starts[0] + capacity))
    length = Counter(gaps.tolist()).most_common(1)[0][0]
    odd = np.flatnonzero(gaps != length)
    if len(odd) != 1:
        raise ValueError(
            f"Expected one gap of a length other than {length} (the seam), found {len(odd)}: {gaps[odd][:10]}."
        )
    return starts, gaps, int(odd[0]), int(length)


def rebuild(memmaps: dict[str, np.memmap], capacity: int, episode_length: int | None) -> dict:
    """The bookkeeping dict, plus "seam" (start, length) and "reset_rewards" for the summary."""
    action, reward = np.asarray(memmaps["action"]), np.asarray(memmaps["reward"])
    starts, gaps, seam_index, length = find_episodes(action, capacity)
    if episode_length is not None and length != episode_length:
        raise ValueError(f"The common episode length is {length}, not {episode_length}.")

    # Storage order: the start right after the seam gap is the oldest episode; the gap's own start is the newest.
    lengths = gaps.copy()
    lengths[seam_index] = length  # The seam gap holds the newest episode and then the seam.
    order = np.roll(np.arange(len(starts)), -(seam_index + 1))
    boundaries = deque((int(starts[i]), int(starts[i] + lengths[i])) for i in order)
    head, tail = boundaries[0][0], boundaries[-1][1] % capacity
    seam_length = int(gaps[seam_index] - length)
    seam_rows = np.arange(tail, tail + seam_length) % capacity
    is_full = bool(seam_length == 0 or np.any(action[seam_rows] != 0) or np.any(reward[seam_rows] != 0))

    episode = {key: [torch.from_numpy(np.array(memmaps[key][0]))] for key in KEYS}  # One fake step from row 0.
    info = EpisodeDatasetInfo(
        fields=Fields.from_episode(episode),
        num_episodes=len(boundaries),
        num_timesteps=int(lengths.sum()),
        episode_lengths=[int(n) for n in lengths[order]],
        is_full=is_full,
    )
    bookkeeping = {"capacity": capacity, "info": info, "head": head, "tail": tail, "episode_boundaries": boundaries}
    return {**bookkeeping, "seam": (tail, seam_length), "reset_rewards": reward[starts[order]]}


def frame_diff(obs: np.memmap, rows: np.ndarray, capacity: int) -> np.ndarray:
    """Mean absolute difference between consecutive frames at the given (circular) rows."""
    frames = obs[rows % capacity].astype(np.float32)
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))


def frame_diagnostics(result: dict, obs: np.memmap, sample_episodes: int, seed: int) -> dict[str, float]:
    """Consecutive-frame differences within episodes, against those across the seam and in the newest episode."""
    capacity, boundaries, (tail, seam_length) = result["capacity"], result["episode_boundaries"], result["seam"]
    generator = np.random.default_rng(seed)
    sampled = generator.choice(len(boundaries), size=min(sample_episodes, len(boundaries)), replace=False)
    diffs = [frame_diff(obs, np.arange(*boundaries[i]), capacity) for i in sampled]
    newest = frame_diff(obs, np.arange(*boundaries[-1]), capacity)
    diagnostics = {
        "within_episode_median": float(np.median(np.concatenate(diffs))),
        "within_episode_max": float(max(diff.max() for diff in diffs)),
        "newest_episode_max": float(newest.max()),
    }
    if seam_length > 0:
        head = boundaries[0][0]
        diagnostics["across_tail"] = float(frame_diff(obs, np.array([tail - 1, tail]), capacity)[0])
        diagnostics["across_head"] = float(frame_diff(obs, np.array([head - 1, head]), capacity)[0])
        if seam_length > 1:
            diagnostics["within_seam_median"] = float(
                np.median(frame_diff(obs, np.arange(tail, tail + seam_length), capacity))
            )
    return diagnostics


def save_atomic(obj: dict, path: str) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, temporary)
    os.replace(temporary, path)


def summarize(result: dict, diagnostics: dict[str, float]) -> None:
    info, capacity, boundaries = result["info"], result["capacity"], result["episode_boundaries"]
    tail, seam_length = result["seam"]
    print(f"capacity: {capacity}, fields: {dict(info.fields)}")
    print(f"episodes: {info.num_episodes}, timesteps: {info.num_timesteps}, is_full: {info.is_full}")
    print(f"episode lengths: {dict(sorted(Counter(info.episode_lengths).items()))}")
    print(
        f"head: {result['head']}, tail: {result['tail']}, oldest: {boundaries[0]}, newest: {boundaries[-1]}"
        f"{' (wraps)' if boundaries[-1][1] > capacity else ''}"
    )
    print(f"seam: {seam_length} rows from {tail} to {(tail + seam_length) % capacity} (exclusive)")
    rewards = result["reset_rewards"]
    all_nan, all_zero = bool(np.isnan(rewards).all()), bool((rewards == 0).all())
    print(f"reset rewards: {len(rewards)} values, all NaN: {all_nan}, all 0: {all_zero}")
    print("frame differences:", ", ".join(f"{key}={value:.2f}" for key, value in diagnostics.items()))


def check(
    result: dict, memmaps: dict[str, np.memmap], path: str, episode_length: int | None, diagnostics: dict
) -> bool:
    failures = []

    def expect(condition: bool, message: str) -> None:
        print(f"[{'ok' if condition else 'FAIL'}] {message}")
        if not condition:
            failures.append(message)

    capacity, info, head, tail, boundaries = (
        result[key] for key in ("capacity", "info", "head", "tail", "episode_boundaries")
    )
    lengths = [end - start for start, end in boundaries]
    expect(
        all(0 <= start < capacity and end > start for start, end in boundaries),
        "boundaries are (start < capacity, end > start)",
    )
    expect(
        all(boundaries[i][1] % capacity == boundaries[i + 1][0] for i in range(len(boundaries) - 1)),
        "boundaries are contiguous modulo capacity",
    )
    expect(
        boundaries[0][0] == head and boundaries[-1][1] % capacity == tail, "head/tail match the oldest/newest episode"
    )
    expect(
        lengths == info.episode_lengths and len(boundaries) == info.num_episodes, "info lengths/count match boundaries"
    )
    expect(sum(lengths) == info.num_timesteps <= capacity, "sum of lengths == num_timesteps <= capacity")
    if episode_length is not None:
        expect(all(n == episode_length for n in lengths), f"every episode has {episode_length} entries")
    expect(
        (head - tail) % capacity == capacity - info.num_timesteps or (head == tail and info.num_timesteps == capacity),
        "the seam has exactly capacity - num_timesteps rows",
    )

    action, reward = np.asarray(memmaps["action"]), np.asarray(memmaps["reward"])
    starts = np.array([start for start, _ in boundaries])
    start_mask = np.zeros(capacity, dtype=bool)
    start_mask[starts] = True
    expect(np.array_equal(np.isnan(action).any(axis=1), start_mask), "NaN actions occur exactly at the episode starts")
    reset_rewards = reward[starts]
    expect(bool(np.isnan(reset_rewards).all() or (reset_rewards == 0).all()), "reset rewards are all NaN or all 0")
    inside = np.zeros(capacity, dtype=bool)
    for start, end in boundaries:
        inside[np.arange(start + 1, end) % capacity] = True
    expect(bool(np.isfinite(reward[inside]).all()), "rewards inside episodes are finite")
    seam_rows = (
        np.arange(tail, head if head > tail else head + capacity) % capacity if head != tail else np.array([], int)
    )
    stale = bool(np.any(action[seam_rows] != 0) or np.any(reward[seam_rows] != 0))
    expect(info.is_full == (stale or len(seam_rows) == 0), "is_full matches whether the seam holds overwritten data")
    if "across_tail" in diagnostics:
        expect(diagnostics["across_tail"] > diagnostics["within_episode_median"], "frames across the tail seam differ")
    expect(
        diagnostics["newest_episode_max"] <= 1.5 * diagnostics["within_episode_max"],
        "newest episode has no unusual frame jump (it was not truncated)",
    )

    loaded = torch.load(path, weights_only=False)
    expect(
        set(loaded) == {"capacity", "info", "head", "tail", "episode_boundaries"}, "saved file has the expected keys"
    )
    expect(
        isinstance(loaded["episode_boundaries"], deque)
        and list(loaded["episode_boundaries"]) == list(boundaries)
        and loaded["info"] == info
        and (loaded["capacity"], loaded["head"], loaded["tail"]) == (capacity, head, tail),
        "saved file round-trips",
    )
    print(f"{len(failures)} failures")
    return not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("buffer_dir")
    parser.add_argument("capacity", type=int)
    parser.add_argument("--output", help="default: BUFFER_DIR/bookkeeping.pt")
    parser.add_argument("--check", action="store_true", help="validate the result and the saved file")
    parser.add_argument("--episode-length", type=int, help="require this common episode length")
    parser.add_argument("--sample-episodes", type=int, default=200, help="episodes sampled for the frame diagnostics")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = args.output or os.path.join(args.buffer_dir, "bookkeeping.pt")

    memmaps = open_memmaps(args.buffer_dir, args.capacity)
    result = rebuild(memmaps, args.capacity, args.episode_length)
    diagnostics = frame_diagnostics(result, memmaps["obs"], args.sample_episodes, args.seed)
    summarize(result, diagnostics)
    save_atomic({key: result[key] for key in ("capacity", "info", "head", "tail", "episode_boundaries")}, output)
    print(f"saved {output}")
    if args.check and not check(result, memmaps, output, args.episode_length, diagnostics):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
