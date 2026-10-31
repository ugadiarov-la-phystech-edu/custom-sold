from collections.abc import Iterator
from sold.data.ring_buffer import RingBufferDataset
from sold.utils.profiling import Timings
import torch


def sample_batches(
    buffer: RingBufferDataset, num_updates: int, timings: Timings | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    """num_updates batches from the replay buffer; none while it is empty.

    They come from a generator seeded with one draw of the global RNG, like the DataLoader worker that used to sample
    was, so that runs keep the same sample stream. timings records each sample as replay_buffer/sample.
    """
    generator = torch.Generator().manual_seed(torch.empty((), dtype=torch.int64).random_().item())
    if buffer.is_empty:
        return
    timings = timings or Timings(enabled=False)
    for _ in range(num_updates):
        with timings.measure("replay_buffer/sample"):
            batch = buffer.sample(generator)
        yield batch
