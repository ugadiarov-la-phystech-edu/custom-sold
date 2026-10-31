from collections import defaultdict
from collections.abc import Mapping
import json
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import os
import re
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video
from torchvision.utils import save_image
from typing import Any


def scalar(value: torch.Tensor | float) -> float:
    """A metric as a float, through the default float dtype like LightningModule.log (45 is written as 45.0)."""
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.get_default_dtype())
    return tensor.item()


class MetricsWriter:
    """Scalars, images and videos of a run: TensorBoard events plus metrics/metrics.jsonl, images/ and videos/."""

    def __init__(self, log_dir: str, experiment: SummaryWriter | None = None) -> None:
        self.log_dir = log_dir
        self.save_dirs = {subdir: os.path.join(log_dir, subdir) for subdir in ["metrics", "images", "videos"]}
        for path in self.save_dirs.values():
            os.makedirs(path, exist_ok=True)
        self.experiment = experiment if experiment is not None else SummaryWriter(log_dir)
        self.current_step = 0
        self.accumulated_metrics: dict[str, float] = {}
        self.metrics_file = open(os.path.join(self.save_dirs["metrics"], "metrics.jsonl"), mode="a")

    @classmethod
    def create(cls, save_dir: str, name: str = "logs") -> "MetricsWriter":
        """A writer for a new save_dir/name/version_N (the layout of Lightning's TensorBoardLogger)."""
        root = os.path.join(save_dir, name)
        existing = os.listdir(root) if os.path.isdir(root) else []
        versions = [int(match.group(1)) for entry in existing if (match := re.fullmatch(r"version_(\d+)", entry))]
        return cls(os.path.join(root, f"version_{max(versions, default=-1) + 1}"))

    def log_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        values = {name: scalar(value) for name, value in metrics.items()}
        for name, value in values.items():
            self.experiment.add_scalar(name, value, step)
        if step > self.current_step:
            self._flush_metrics()
        self.current_step = step
        self.accumulated_metrics.update(values)

    def finalize(self) -> None:
        self._flush_metrics()  # Otherwise the metrics of the last step never reach metrics.jsonl.
        self.experiment.flush()

    def _flush_metrics(self) -> None:
        if self.accumulated_metrics:
            record = {"step": self.current_step, **self.accumulated_metrics}
            self.metrics_file.write(json.dumps(record) + "\n")
            self.metrics_file.flush()
            self.accumulated_metrics.clear()

    def log_image(self, name: str, image: torch.Tensor, step: int) -> None:
        self.experiment.add_image(name, image, step)
        save_image(image, os.path.join(self.save_dirs["images"], name) + f"-step={step}.png")

    def log_video(self, name: str, video: torch.Tensor, step: int, fps: int = 10) -> None:
        if video.dtype != torch.uint8:
            video = (video * 255).to(torch.uint8)

        self.experiment.add_video(name, np.expand_dims(video.cpu().numpy(), 0), global_step=step)
        name = name.replace("/", "_")  # Turn tensorboard grouping into valid file name.
        write_video(os.path.join(self.save_dirs["videos"], name) + f"-step={step}.mp4", video.permute(0, 2, 3, 1), fps)


class LoggingStepMixin:
    """Log against a custom 'logging_step' instead of Lightning's global step.

    With an ExtendedTensorBoardLogger, scalars are written once per logging step (after the epoch's last batch, at most
    every max_logging_freq steps); images and videos immediately.
    """

    def __init__(self, max_logging_freq: int = 1) -> None:
        super().__init__()
        self.max_logging_freq = max_logging_freq
        self.last_logging_step = defaultdict(lambda: -max_logging_freq)

    @property
    def logging_step(self) -> int:
        return self.current_epoch

    def is_logging_step(self, name: str) -> bool:
        """Whether a scalar logged under name now would be written."""
        return self.trainer.is_last_batch and self.logging_step - self.last_logging_step[name] >= self.max_logging_freq

    def log(self, name: str, value: torch.Tensor | float, *args, **kwargs) -> None:
        """Log a scalar; tensors become numbers only when written, not on every call."""
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            if kwargs.get("logger", True) and self.is_logging_step(name):
                self.last_logging_step[name] = self.logging_step
                self.logger.log_metrics({name: scalar(value)}, step=self.logging_step)
            kwargs["logger"] = False  # Written above; Lightning still gets it for the progress bar and callbacks.
        super().log(name, value, *args, **kwargs)

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """Log an image (3, height, width) in [0, 1] (ExtendedTensorBoardLogger only)."""
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            self.logger.log_image(name, image, step=self.logging_step)

    def log_video(self, name: str, video: torch.Tensor) -> None:
        """Log a video (time, 3, height, width) (ExtendedTensorBoardLogger only)."""
        if isinstance(self.logger, ExtendedTensorBoardLogger):
            self.logger.log_video(name, video, step=self.logging_step)


class ExtendedTensorBoardLogger(TensorBoardLogger):
    """Lightning's TensorBoard logger (SAVi training) with a MetricsWriter's metrics.jsonl, images/ and videos/."""

    def __init__(
        self,
        save_dir: _PATH,
        name: str | None = "lightning_logs",
        version: int | str | None = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: _PATH | None = None,
        **kwargs: Any,
    ):
        super().__init__(save_dir, name, version, log_graph, default_hp_metric, prefix, sub_dir, **kwargs)
        self.writer = MetricsWriter(self.log_dir, self.experiment)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        self.writer.log_metrics(metrics, step)

    def finalize(self, status: str) -> None:
        self.writer.finalize()
        super().finalize(status)

    def log_image(self, name: str, image: torch.Tensor, step: int) -> None:
        self.writer.log_image(name, image, step)

    def log_video(self, name: str, video: torch.Tensor, step: int, fps: int = 10) -> None:
        self.writer.log_video(name, video, step, fps)
