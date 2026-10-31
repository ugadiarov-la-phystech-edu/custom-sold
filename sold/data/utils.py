from collections.abc import Iterator
from sold.data.ring_buffer import RingBufferDataset
import time
import torch
from torch.utils.data import IterableDataset


class NumUpdatesWrapper(IterableDataset):
    def __init__(self, dataset: RingBufferDataset, num_updates: int, measure_sampling: bool = False) -> None:
        self.dataset = dataset
        self.num_updates = num_updates
        self.measure_sampling = measure_sampling  # Add "sample_seconds" to each sample.

    def __iter__(self) -> Iterator:
        if self.dataset.is_empty:
            return iter([])

        for _ in range(self.num_updates):
            start_time = time.perf_counter()
            sample = self.dataset.sample()
            if self.measure_sampling:
                sample["sample_seconds"] = torch.tensor(time.perf_counter() - start_time, dtype=torch.float64)
            yield sample
