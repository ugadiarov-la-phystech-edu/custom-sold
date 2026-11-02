"""Drift test: compare the training updates of models that should train alike (e.g. eager and compiled), update by
update.

lockstep() runs one training step per model on the same batch and seed and compares each model with the first one:
every training stage's outputs (stage/...), the clipped gradients of each optimizer (grad/...), the update of each
module's parameters (update/...), the parameters and buffers after the update (state/..., buffer/...). With resync,
each other model gets the first one's training state before every step, so each record holds that step's own error;
without, the models train freely and the errors accumulate.

Updates are compared on the elements whose gradient is not zero in theory: Adam turns rounding noise in such gradients
(e.g. of attention key biases) into steps of about the learning rate.
"""

from collections.abc import Iterable, Iterator, Mapping
import math
import numpy as np
import torch
from torch import nn
from typing import Any

# Elements whose reference gradient is below this fraction of the largest one count as zero in theory.
ZERO_GRADIENT = 1e-5


def copy_training_state(source: nn.Module, target: nn.Module) -> None:
    """Copy parameters, buffers and optimizer states from source to target, in place: compiled CUDA graphs keep reading
    the target's memory."""
    target.load_state_dict(source.state_dict())  # Copies into the existing tensors.
    for source_optimizer, target_optimizer in zip(source.optimizers, target.optimizers, strict=True):
        for source_param, target_param in zip(_params(source_optimizer), _params(target_optimizer), strict=True):
            target_state = target_optimizer.state[target_param]
            for key, value in source_optimizer.state.get(source_param, {}).items():
                if key in target_state:
                    target_state[key].copy_(value)
                else:
                    target_state[key] = value.clone()


def _params(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [param for group in optimizer.param_groups for param in group["params"]]


def _flatten(outputs: Mapping[str, Any], prefix: str) -> dict[str, torch.Tensor]:
    flat = {}
    for key, value in outputs.items():
        if isinstance(value, Mapping):
            flat |= _flatten(value, f"{prefix}{key}.")
        elif isinstance(value, torch.Tensor) and value.is_floating_point():
            flat[f"{prefix}{key}"] = value.detach().clone()
    return flat


class Recorder:
    """Records a model's training step: stage outputs, clipped gradients and parameter updates.

    Wraps the model's _<stage>_stage methods (after compile_stages, so the compiled versions run inside the wrappers)
    and hooks its optimizers' steps.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.outputs: dict[str, torch.Tensor] = {}
        self.gradients: dict[str, torch.Tensor] = {}  # Parameter name -> its gradient when the optimizer stepped.
        self.before: dict[str, torch.Tensor] = {}
        self.names = {param: name for name, param in model.named_parameters()}
        for stage in model.STAGES:
            name = f"_{stage}_stage"
            setattr(model, name, self._wrap(stage, getattr(model, name)))
        for optimizer in model.optimizers:
            optimizer.register_step_pre_hook(self._record_gradients)

    def _wrap(self, stage: str, method):
        def recorded(*args, **kwargs):
            outputs = method(*args, **kwargs)
            self.outputs |= _flatten(outputs, f"stage/{stage}/")
            return outputs

        return recorded

    def _record_gradients(self, optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        for param in _params(optimizer):
            if param.grad is not None:
                self.gradients[self.names[param]] = param.grad.detach().clone()

    def start(self) -> None:
        self.outputs, self.gradients = {}, {}
        self.before = {name: param.detach().clone() for name, param in self.model.named_parameters()}

    def updates(self) -> dict[str, torch.Tensor]:
        """Each parameter's change since start()."""
        return {name: param.detach() - self.before[name] for name, param in self.model.named_parameters()}


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """||actual - expected|| / ||expected||; 0 when both are zero. NaNs count as zeros."""
    actual, expected = (torch.nan_to_num(x.detach().double(), nan=0.0) for x in (actual, expected))
    difference = (actual - expected).norm().item()
    norm = expected.norm().item()
    return difference / norm if norm > 0 else (0.0 if difference == 0 else math.inf)


def _module_of(name: str) -> str:
    return name.split(".")[0]


def _grouped(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Parameter tensors concatenated per top-level module."""
    groups: dict[str, list[torch.Tensor]] = {}
    for name, tensor in tensors.items():
        groups.setdefault(_module_of(name), []).append(tensor.flatten())
    return {module: torch.cat(parts) for module, parts in groups.items()}


def _nonzero_masks(
    updates: Mapping[str, torch.Tensor], gradients: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Per module, the elements whose gradient is not zero in theory (all elements of parameters without gradients)."""
    maxima: dict[str, float] = {}
    for name, gradient in gradients.items():
        maxima[_module_of(name)] = max(maxima.get(_module_of(name), 0.0), gradient.abs().max().item())
    masks = {
        name: gradients[name].abs() > ZERO_GRADIENT * maxima[_module_of(name)]
        if name in gradients
        else torch.ones_like(update, dtype=torch.bool)
        for name, update in updates.items()
    }
    return _grouped(masks)


def compare(reference: Recorder, other: Recorder) -> dict[str, float]:
    """Relative errors of other's last recorded step against reference's."""
    errors = {key: relative_error(other.outputs[key], value) for key, value in reference.outputs.items()}
    reference_gradients, other_gradients = _grouped(reference.gradients), _grouped(other.gradients)
    errors |= {f"grad/{m}": relative_error(other_gradients[m], g) for m, g in reference_gradients.items()}
    updates = reference.updates()
    masks = _nonzero_masks(updates, reference.gradients)
    reference_updates, other_updates = _grouped(updates), _grouped(other.updates())
    for module, update in reference_updates.items():
        errors[f"update/{module}"] = relative_error(other_updates[module][masks[module]], update[masks[module]])
    reference_params = _grouped(dict(reference.model.named_parameters()))
    other_params = _grouped(dict(other.model.named_parameters()))
    errors |= {f"state/{m}": relative_error(other_params[m], params) for m, params in reference_params.items()}
    reference_state, other_state = reference.model.state_dict(), other.model.state_dict()
    for name, _ in reference.model.named_buffers():
        if name in reference_state:
            errors[f"buffer/{name}"] = relative_error(other_state[name], reference_state[name])
    return errors


def scalars(recorder: Recorder) -> dict[str, float]:
    """The mean of every stage output and the value of every one-element buffer, e.g. the return percentiles."""
    values = {key: value.nan_to_num().mean().item() for key, value in recorder.outputs.items()}
    values |= {f"buffer/{name}": b.item() for name, b in recorder.model.named_buffers() if b.numel() == 1}
    return values


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)  # Also seeds every CUDA device.


def lockstep(
    models: Mapping[str, nn.Module],
    batches: Iterable[dict[str, torch.Tensor]],
    resync: bool = True,
    seed_offsets: Mapping[str, int] | None = None,
) -> Iterator[dict[str, float]]:
    """One record per batch: each model's scalars ("<name>/<key>") and each model's errors against the first one
    ("<name>/err/<key>").

    Models need their optimizers (model.optimizers) and compiled stages before the first step. Step k runs every model
    with seed k + seed_offsets[name] (0 by default).
    """
    seed_offsets = seed_offsets or {}
    recorders = {name: Recorder(model) for name, model in models.items()}
    (reference_name, reference), *others = recorders.items()
    for step, batch in enumerate(batches):
        if resync:
            for _, other in others:
                copy_training_state(reference.model, other.model)
        for name, recorder in recorders.items():
            recorder.start()
            seed_everything(step + seed_offsets.get(name, 0))
            if recorder.model.cudagraphs:
                torch.compiler.cudagraph_mark_step_begin()
            recorder.model.training_step(batch)
        record = {"step": float(step)}
        for name, recorder in recorders.items():
            record |= {f"{name}/{key}": value for key, value in scalars(recorder).items()}
        for name, other in others:
            record |= {f"{name}/err/{key}": value for key, value in compare(reference, other).items()}
        yield record


def summarize(
    records: list[dict[str, float]], model: str, floor_model: str | None = None, threshold: float = 1e-2
) -> dict[str, dict[str, Any]]:
    """Per error metric of model: median, p99, max, the step of the max, and whether it is flagged.

    Flagged: the max exceeds threshold and 10 times the floor (floor_model's max for the metric, e.g. an eager model
    run twice), or the errors grow: the last quarter's median is more than 10 times the first quarter's and above
    threshold / 10 (under re-sync, growth means state that the re-sync does not reach).
    """
    prefix = f"{model}/err/"
    summary = {}
    for key in sorted(k for k in records[0] if k.startswith(prefix)):
        metric = key[len(prefix) :]
        values = np.array([record[key] for record in records])
        floor = max(record[f"{floor_model}/err/{metric}"] for record in records) if floor_model else 0.0
        quarter = max(1, len(values) // 4)
        first, last = np.median(values[:quarter]), np.median(values[-quarter:])
        too_large = values.max() > max(threshold, 10 * floor)
        grows = last > threshold / 10 and last > 10 * first
        summary[metric] = {
            "median": float(np.median(values)),
            "p99": float(np.percentile(values, 99)),
            "max": float(values.max()),
            "step_of_max": int(records[int(values.argmax())]["step"]),
            "floor": floor,
            "flagged": bool(too_large or grows),
        }
    return summary


def bias(records: list[dict[str, float]], key: str, model: str, reference: str) -> dict[str, float]:
    """Mean difference of model's scalar key from reference's over the records, with its standard error (corrected
    for lag-1 autocorrelation) and z = mean / standard error."""
    differences = np.array([record[f"{model}/{key}"] - record[f"{reference}/{key}"] for record in records])
    n = len(differences)
    centered = differences - differences.mean()
    variance = centered.var()
    rho = float((centered[1:] * centered[:-1]).mean() / variance) if variance > 0 and n > 2 else 0.0
    effective_n = n * (1 - rho) / (1 + rho) if -1 < rho < 1 else 1.0
    standard_error = math.sqrt(variance / max(effective_n, 1.0))
    mean = float(differences.mean())
    return {"mean": mean, "standard_error": standard_error, "z": mean / standard_error if standard_error > 0 else 0.0}
