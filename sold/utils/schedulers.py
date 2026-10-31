import math
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import List


class LinearWarmupCosineAnnealingLR(LRScheduler):
    """Linear warmup from warmup_start_lr over warmup_epochs, then cosine annealing to eta_min at max_epochs.

    Copied from lightning-bolts 0.7.0 (Apache License 2.0) with the same arithmetic: the learning rates match exactly.
    """

    def __init__(self, optimizer: Optimizer, warmup_epochs: int, max_epochs: int, warmup_start_lr: float = 0.0,
                 eta_min: float = 0.0, last_epoch: int = -1) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """The chainable (recursive) form, as in lightning-bolts."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, please use `get_last_lr()`.")

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        cosine_epochs = self.max_epochs - self.warmup_epochs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * cosine_epochs) == 0:
            return [group["lr"] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / cosine_epochs)) / 2
                    for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)]

        return [(1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / cosine_epochs))
                / (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / cosine_epochs))
                * (group["lr"] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]
