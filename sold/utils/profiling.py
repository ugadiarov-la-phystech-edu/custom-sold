from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import time
import torch


class Timings:
    """Mean milliseconds per call of named sections: wall clock, or CUDA events (cuda=True) read in summary()."""

    def __init__(self, enabled: bool = True, event_factory: Callable[..., torch.cuda.Event] = torch.cuda.Event) -> None:
        self.enabled = enabled
        self.event_factory = event_factory
        self.seconds: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def measure(self, name: str, cuda: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield
        elif cuda:
            start, end = self.event_factory(enable_timing=True), self.event_factory(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events.append((name, start, end))
        else:
            start_time = time.perf_counter()
            yield
            self.add(name, time.perf_counter() - start_time)

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.seconds[name] += seconds
            self.counts[name] += 1

    def summary(self) -> dict[str, float]:
        """Mean milliseconds per section since the last summary."""
        if self.events:
            self.events[-1][2].synchronize()
        for name, start, end in self.events:
            self.add(name, start.elapsed_time(end) / 1000)
        means = {name: 1000 * self.seconds[name] / self.counts[name] for name in self.counts}
        self.seconds.clear()
        self.counts.clear()
        self.events.clear()
        return means
