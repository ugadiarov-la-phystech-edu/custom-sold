"""utils/profiling.py: section timings, by the wall clock or by CUDA events."""

from sold.utils import profiling
from sold.utils.profiling import Timings


def test_wall_clock_sections_report_the_mean_milliseconds(monkeypatch):
    clock = iter([1.0, 1.5, 2.0, 2.25])
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(clock))
    timings = Timings()
    for _ in range(2):
        with timings.measure("collect/env_step"):
            pass
    timings.add("replay_buffer/sample", 0.004)
    assert timings.summary() == {"collect/env_step": 375.0, "replay_buffer/sample": 4.0}
    assert timings.summary() == {}


class FakeEvent:
    clock = 0.0

    def __init__(self, enable_timing):
        assert enable_timing
        self.synchronized = False

    def record(self):
        FakeEvent.clock += 5.0
        self.time = FakeEvent.clock

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, end):
        assert end.synchronized
        return end.time - self.time


def test_cuda_sections_are_read_from_events_in_the_summary():
    timings = Timings(event_factory=FakeEvent)
    with timings.measure("learn/imagination", cuda=True):
        pass
    assert timings.counts == {}
    assert timings.summary() == {"learn/imagination": 5.0}


def test_disabled_timings_record_nothing():
    timings = Timings(enabled=False)
    with timings.measure("collect/env_step"), timings.measure("learn/imagination", cuda=True):
        pass
    timings.add("replay_buffer/sample", 1.0)
    assert timings.summary() == {}
