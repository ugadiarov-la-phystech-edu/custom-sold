from collections.abc import Iterable
import torch.nn as nn


class FreezeParameters:
    def __init__(self, modules: Iterable[nn.Module]):
        """Freeze the parameters of a list of nn.Modules."""
        self.parameters = [param for module in modules for param in module.parameters()]
        self.initial_param_states = [param.requires_grad for param in self.parameters]

    def __enter__(self) -> None:
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        for param, requires_grad in zip(self.parameters, self.initial_param_states):
            param.requires_grad = requires_grad
        return False
