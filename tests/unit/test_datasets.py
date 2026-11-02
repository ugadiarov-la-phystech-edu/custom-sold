"""datasets: field descriptions, the replay ring buffer, sample_batches and SAVi image datasets."""

from collections import deque
import copy
import logging
import numpy as np
import pytest
from sold.data.image import ImageDataset, NPZDataset, PNGDataset, load_image_dataset
from sold.data.info import ColorImage, EpisodeDatasetInfo, Fields, FieldType, Scalar, TensorType, Vector
from sold.data.ring_buffer import EpisodeDataset, RingBufferDataset
from sold.data.utils import sample_batches
from tests.conftest import add_episode, make_episode, write_image_dataset
import torch


# ------------------------------------------------------------------------------------- info
def test_tensor_type_converts_strings():
    field = TensorType(device="cpu", dtype="float16", shape=[2, 3])
    assert field.device == torch.device("cpu") and field.dtype == torch.float16 and field.shape == torch.Size([2, 3])
    assert field.element_size() == 2 and field.step_size() == 12
    assert TensorType.is_compatible(torch.zeros(1)) and not TensorType.is_compatible(np.zeros(1))
    assert FieldType.is_compatible(None) is None  # Abstract body.


def test_scalar_and_vector_validation():
    assert Scalar(device="cpu", dtype=torch.float32, shape=()).step_size() == 4
    with pytest.raises(ValueError, match="zero-dimensional"):
        Scalar(device="cpu", dtype=torch.float32, shape=(1,))
    assert Vector(device="cpu", dtype=torch.float32, shape=(3,)).step_size() == 12
    with pytest.raises(ValueError, match="one-dimensional"):
        Vector(device="cpu", dtype=torch.float32, shape=(1, 2))


def test_color_image_is_height_width_channels_uint8():
    image = ColorImage(device="cpu", height=4, width=5)
    assert image.dtype == torch.uint8 and image.shape == (4, 5, 3) and image.step_size() == 60
    assert ColorImage.is_compatible(torch.zeros(4, 5, 3, dtype=torch.uint8))
    # Channel-first frames, as stored by SOLD, are not ColorImages.
    assert not ColorImage.is_compatible(torch.zeros(3, 4, 5, dtype=torch.uint8))


def test_fields_from_episode_infers_types():
    episode = {
        "reward": [torch.tensor(1.0)],
        "action": [torch.zeros(2)],
        "hwc": [torch.zeros(4, 5, 3, dtype=torch.uint8)],
        "obs": [torch.zeros(3, 4, 5, dtype=torch.uint8)],
    }
    fields = Fields.from_episode(episode)
    assert type(fields["reward"]) is Scalar and type(fields["action"]) is Vector
    assert type(fields["hwc"]) is ColorImage and type(fields["obs"]) is TensorType
    assert fields["obs"].shape == (3, 4, 5)


@pytest.mark.parametrize(
    "episode, match",
    [
        ({"x": []}, "to be a list"),
        ({"x": (torch.zeros(1),)}, "to be a list"),
        ({"x": [np.zeros(1)]}, "Unsupported type"),
    ],
)
def test_fields_from_episode_errors(episode, match):
    with pytest.raises(ValueError, match=match):
        Fields.from_episode(episode)


def test_episode_dataset_info_defaults():
    info = EpisodeDatasetInfo()
    assert info.is_empty and info.episode_lengths == [] and not info.is_full
    assert EpisodeDatasetInfo(num_episodes=1).is_empty is False


# ------------------------------------------------------------------------------ ring buffer
FILES = ["action.dat", "bookkeeping.pt", "obs.dat", "reward.dat"]


@pytest.fixture
def buffer(tmp_path):
    return RingBufferDataset(capacity=10, batch_size=4, sequence_length=3, save_path=str(tmp_path / "rb"))


def test_ring_buffer_starts_empty(buffer, recwarn):
    from sold.data.info import EpisodeDatasetInfoMixin

    assert buffer.is_empty and len(buffer) == 0 and buffer.fields is None
    assert EpisodeDatasetInfoMixin.is_empty.fget(buffer)  # Overridden by RingBufferDataset.
    assert buffer.average_episode_length == 0 and buffer.last_episode_return is None
    assert "fields: None" in repr(buffer)


def test_ring_buffer_add_step_converts_non_tensors(buffer):
    buffer.add_step({"obs": np.zeros((3, 2, 2), np.uint8), "reward": 1.5})
    assert isinstance(buffer.current_episode["obs"][0], torch.Tensor)
    assert buffer.current_episode["reward"][0].item() == 1.5
    assert buffer.num_episodes == 0


def test_ring_buffer_stores_episode_and_statistics(buffer):
    add_episode(buffer, make_episode(4))
    assert (buffer.num_episodes, buffer.num_timesteps, buffer.episode_lengths) == (1, 4, [4])
    assert buffer.last_episode_return.item() == pytest.approx(1 + 2 + 3)
    assert buffer.ring_buffer["obs"][:4, 0, 0, 0].tolist() == [0, 1, 2, 3]
    assert (buffer.head, buffer.tail, list(buffer.episode_boundaries)) == (0, 4, [(0, 4)])
    assert "num_episodes: 1" in repr(buffer) and "['obs', 'action', 'reward']" in repr(buffer)
    assert buffer.average_episode_length == 4 and not buffer.is_empty and not buffer.is_full


def test_ring_buffer_keeps_one_unfinished_episode_per_stream(tmp_path):
    buffer = RingBufferDataset(100, 1, 2, save_path=str(tmp_path))
    for t in range(3):  # Two environments' episodes, stepped alternately.
        for stream in (0, 1):
            step = {"obs": torch.full((3, 2, 2), 10 * stream + t, dtype=torch.uint8), "reward": float(t)}
            buffer.add_step(step, done=t == 2, stream=stream)
    assert buffer.num_episodes == 2 and buffer.episode_lengths == [3, 3] and not buffer.current_episodes
    stored = [buffer.ring_buffer["obs"][start:end, 0, 0, 0].tolist() for start, end in buffer.episode_boundaries]
    assert stored == [[0, 1, 2], [10, 11, 12]]  # Each episode contiguous, from its own stream.


def test_ring_buffer_wraps_and_evicts_oldest(buffer):
    add_episode(buffer, make_episode(4, start=0))
    add_episode(buffer, make_episode(4, start=10))
    add_episode(buffer, make_episode(4, start=20))  # Needs eviction of the first episode and wraps around.
    assert buffer.num_episodes == 2 and buffer.num_timesteps == 8 and buffer.is_full
    assert list(buffer.episode_boundaries) == [(4, 8), (8, 12)]  # Boundaries are stored unwrapped.
    assert buffer.head == 4 and buffer.tail == 2
    assert buffer.ring_buffer["obs"][[8, 9, 0, 1], 0, 0, 0].tolist() == [20, 21, 22, 23]


def test_ring_buffer_remove_last_episode_sets_head_to_tail(buffer):
    add_episode(buffer, make_episode(4))
    buffer.remove_episode()
    assert buffer.head == buffer.tail == 4 and len(buffer.episode_boundaries) == 0


def test_ring_buffer_sample_returns_contiguous_sequences(buffer):
    add_episode(buffer, make_episode(4, start=0))
    add_episode(buffer, make_episode(4, start=10))
    add_episode(buffer, make_episode(4, start=20))
    for _ in range(20):
        batch = buffer.sample()
        assert batch["obs"].shape == (4, 3, 3, 2, 2) and batch["action"].shape == (4, 3, 2)
        assert batch["obs"].dtype == torch.uint8
        first = batch["obs"][:, :, 0, 0, 0].long()
        assert torch.all(first[:, 1:] - first[:, :-1] == 1)
        assert set(first[:, 0].tolist()) <= {10, 11, 20, 21}  # Any window that fits, including the last (#19).


def test_ring_buffer_sequence_length_none_samples_all_episodes(tmp_path):
    buffer = RingBufferDataset(capacity=10, batch_size=2, sequence_length=2, save_path=str(tmp_path))
    add_episode(buffer, make_episode(3))
    buffer.sequence_length = None
    with pytest.raises(TypeError):  # sequence_length=None is half-supported: sampling cannot build windows.
        buffer.sample()


def test_ring_buffer_is_empty_warns_when_episodes_are_too_short(buffer):
    add_episode(buffer, make_episode(2))
    with pytest.warns(UserWarning, match="none are of sufficient length"):
        assert buffer.is_empty


def test_ring_buffer_files_and_load_from_disk_storage(buffer, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="sold.data.ring_buffer"):
        add_episode(buffer, make_episode(4))
    assert "Total estimated replay buffer size (GB)" in caplog.text
    buffer.ring_buffer["obs"].flush()
    assert sorted(p.name for p in (tmp_path / "rb").iterdir()) == FILES

    restored = RingBufferDataset(capacity=1, batch_size=4, sequence_length=3, save_path=str(tmp_path / "copy"))
    restored.load_from_disk_storage(
        str(tmp_path / "rb"),
        {
            "capacity": buffer.capacity,
            "info": copy.deepcopy(buffer.info),
            "head": buffer.head,
            "tail": buffer.tail,
            "episode_boundaries": deque(buffer.episode_boundaries),
        },
    )
    assert restored.capacity == 10 and restored.num_episodes == 1 and restored.tail == 4
    assert np.array_equal(restored.ring_buffer["obs"], buffer.ring_buffer["obs"])
    assert restored.sample()["obs"].shape == (4, 3, 3, 2, 2)


def test_ring_buffer_bookkeeping_round_trip(buffer, tmp_path):
    for start in (0, 10, 20):  # The third episode evicts the first and wraps around.
        add_episode(buffer, make_episode(4, start=start))
    buffer.save_bookkeeping()
    assert sorted(p.name for p in (tmp_path / "rb").iterdir()) == FILES
    loaded = RingBufferDataset.load_bookkeeping(tmp_path / "rb")
    assert set(loaded) == {"capacity", "info", "head", "tail", "episode_boundaries"}
    assert (loaded["capacity"], loaded["head"], loaded["tail"]) == (10, 4, 2)
    assert type(loaded["episode_boundaries"]) is deque and list(loaded["episode_boundaries"]) == [(4, 8), (8, 12)]
    assert loaded["info"] == buffer.info and loaded["info"].is_full and loaded["info"].fields == buffer.fields


def test_ring_buffer_bookkeeping_describes_the_episodes_stored_before_the_newest(buffer, tmp_path):
    """add_episode writes the file before storing: if the process dies mid-write, it describes only complete data."""

    def load():
        return RingBufferDataset.load_bookkeeping(tmp_path / "rb")

    add_episode(buffer, make_episode(4, start=0))
    assert load()["info"].num_episodes == 0 and load()["info"].fields == buffer.fields
    add_episode(buffer, make_episode(4, start=10))
    assert (load()["info"].num_episodes, list(load()["episode_boundaries"])) == (1, [(0, 4)])
    add_episode(buffer, make_episode(4, start=20))  # Evicts the first episode before writing the file.
    bookkeeping = load()
    assert (bookkeeping["head"], bookkeeping["tail"], list(bookkeeping["episode_boundaries"])) == (4, 8, [(4, 8)])
    assert bookkeeping["info"].num_episodes == 1 and bookkeeping["info"].is_full

    # A buffer "resumed" after the third episode's write was lost: the second episode is intact.
    buffer.ring_buffer["obs"].flush()
    restored = RingBufferDataset(capacity=10, batch_size=4, sequence_length=3, save_path=str(tmp_path / "copy"))
    restored.load_from_disk_storage(tmp_path / "rb")  # Bookkeeping read from the file.
    assert (restored.num_episodes, restored.head, restored.tail) == (1, 4, 8)
    assert restored.ring_buffer["obs"][4:8, 0, 0, 0].tolist() == [10, 11, 12, 13]


def test_ring_buffer_bookkeeping_is_written_atomically(buffer, tmp_path, monkeypatch):
    add_episode(buffer, make_episode(4))
    buffer.save_bookkeeping()

    def failing_save(obj, path):
        open(path, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        add_episode(buffer, make_episode(4, start=10))
    assert RingBufferDataset.load_bookkeeping(tmp_path / "rb")["info"].num_episodes == 1  # Unchanged.


def test_episode_dataset_initializes_storage_from_given_fields():
    initialized = []

    class Recording(EpisodeDataset):
        store_episode = remove_episode = sample = lambda self, *args: None

        def _initialize_from_fields(self, fields):
            initialized.append(fields)

    fields = Fields(obs=TensorType(device="cpu", dtype=torch.uint8, shape=(3, 2, 2)))
    Recording(5, 1, 1, info=EpisodeDatasetInfo(fields=fields))
    assert initialized == [fields]


def test_episode_dataset_is_abstract(tmp_path):
    with pytest.raises(TypeError):
        EpisodeDataset(5, 1)

    class Minimal(EpisodeDataset):
        store_episode = remove_episode = sample = lambda self, *args: None

    dataset = Minimal(5, 1)
    dataset._initialize_from_fields(None)
    dataset._before_store()  # No-op hooks.
    for method in (EpisodeDataset.store_episode, EpisodeDataset.remove_episode):
        with pytest.raises(NotImplementedError):
            method(dataset, *([{}] if method is EpisodeDataset.store_episode else []))
    with pytest.raises(NotImplementedError):
        EpisodeDataset.sample(dataset)


def test_sample_batches(buffer):
    assert list(sample_batches(buffer, 3)) == []
    add_episode(buffer, make_episode(4))
    batches = list(sample_batches(buffer, 3))
    assert len(batches) == 3 and batches[0]["obs"].shape == (4, 3, 3, 2, 2)


def test_sample_batches_draw_one_seed_per_round(buffer):
    add_episode(buffer, make_episode(4))
    torch.manual_seed(1)
    first = [batch["obs"] for batch in sample_batches(buffer, 2)]
    torch.manual_seed(1)
    torch.empty((), dtype=torch.int64).random_()  # The one draw of the global RNG per round, as a worker's seed was.
    state = torch.get_rng_state()
    torch.manual_seed(1)
    assert all(torch.equal(a, b) for a, b in zip(first, (batch["obs"] for batch in sample_batches(buffer, 2))))
    assert torch.equal(torch.get_rng_state(), state)


# BUGS_AND_PITFALLS.md #19 (fixed).
def test_replay_buffer_can_sample_the_last_window(tmp_path):
    buffer = RingBufferDataset(capacity=10, batch_size=64, sequence_length=3, save_path=str(tmp_path))
    add_episode(buffer, make_episode(4))
    starts = set(buffer.sample()["obs"][:, 0, 0, 0, 0].tolist())
    assert starts == {0, 1}


# BUGS_AND_PITFALLS.md #25 (fixed).
def test_ring_buffer_accepts_predefined_fields(tmp_path):
    fields = Fields(obs=TensorType(device="cpu", dtype=torch.uint8, shape=(3, 2, 2)))
    buffer = RingBufferDataset(5, 1, 1, info=EpisodeDatasetInfo(fields=fields), save_path=str(tmp_path))
    assert buffer.ring_buffer["obs"].shape == (5, 3, 2, 2)


# ---------------------------------------------------------------------------- image datasets
@pytest.mark.parametrize("save_format, expected_class", [("png", PNGDataset), ("npz", NPZDataset)])
def test_load_image_dataset_detects_format(tmp_path, save_format, expected_class):
    write_image_dataset(tmp_path / "data" / "run", save_format)
    (tmp_path / "data" / "run" / "train" / "not_an_episode").mkdir()
    dataset = load_image_dataset(str(tmp_path / "data"), "run", "train", sequence_length=4, batch_size=2)
    assert type(dataset) is expected_class
    assert len(dataset) == 3 and dataset.episode_dir_names == [0, 1, 2]
    assert dataset.dataset_infos == {"image_size": [16, 16], "action_dim": 2}
    assert dataset.dataset_infos["action_dim"] == 2  # Cached.


@pytest.mark.parametrize("save_format", ["png", "npz"])
def test_image_dataset_items(tmp_path, save_format):
    root = write_image_dataset(tmp_path / "run", save_format, episode_length=7)
    expected = np.load(root / "val" / "1" / "episode.npz")
    val = load_image_dataset(str(tmp_path), "run", "val", sequence_length=4)
    images, actions = val[1]
    assert images.shape == (4, 3, 16, 16) and images.dtype == torch.float32
    assert 0.0 <= images.min() and images.max() <= 1.0
    # Aligned like SOLD's replay buffer: entry t is the action that led to frame t, NaN for frame 0 (#13).
    assert torch.isnan(actions[0]).all()
    assert torch.allclose(actions[1:], torch.from_numpy(expected["actions"][:3]))
    if save_format == "npz":
        assert torch.allclose(images, torch.from_numpy(expected["images"][:4]).permute(0, 3, 1, 2) / 255.0)

    train = load_image_dataset(str(tmp_path), "run", "train", sequence_length=4)
    starts = set()
    train_actions = np.load(root / "train" / "0" / "episode.npz")["actions"]
    for seed in range(40):
        np.random.seed(seed)
        _, actions = train[0]
        assert actions.shape == (4, 2)
        start = (
            0
            if torch.isnan(actions[0]).all()
            else next(s for s in range(1, 4) if np.allclose(train_actions[s - 1 : s + 3], actions.numpy()))
        )
        starts.add(start)
    assert starts == {0, 1, 2, 3}  # Every window, including the last one (#14).


def test_image_dataset_base_class(tmp_path):
    write_image_dataset(tmp_path / "run", "npz")
    dataset = ImageDataset(str(tmp_path), "run", "train", 4)
    images, actions, rewards = dataset._load_npz_data(0)
    assert images.shape == (7, 16, 16, 3) and actions.shape == (6, 2) and rewards.shape == (6,)


# BUGS_AND_PITFALLS.md #14 (fixed).
def test_episode_of_exactly_sequence_length_can_be_loaded(tmp_path):
    write_image_dataset(tmp_path / "run", "npz", episode_length=4)
    images, _ = load_image_dataset(str(tmp_path), "run", "train", sequence_length=4)[0]
    assert images.shape[0] == 4


# BUGS_AND_PITFALLS.md #14 (fixed).
def test_random_crop_can_use_the_last_window(tmp_path):
    write_image_dataset(tmp_path / "run", "npz", episode_length=7)
    dataset = load_image_dataset(str(tmp_path), "run", "train", sequence_length=4)
    images = np.load(tmp_path / "run" / "train" / "0" / "episode.npz")["images"]
    last_window = torch.from_numpy(images[3:7]).permute(0, 3, 1, 2) / 255.0
    assert any(torch.allclose(dataset[0][0], last_window) for _ in range(200))
