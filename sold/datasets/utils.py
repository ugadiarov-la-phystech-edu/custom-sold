from collections.abc import Iterator
from datasets.ring_buffer import RingBufferDataset
from torch.utils.data import IterableDataset


class NumUpdatesWrapper(IterableDataset):
    def __init__(self, dataset: RingBufferDataset, num_updates: int) -> None:
        self.dataset = dataset
        self.num_updates = num_updates

    def __iter__(self) -> Iterator:
        if self.dataset.is_empty:
            return iter([])

        for _ in range(self.num_updates):
            yield self.dataset.sample()
