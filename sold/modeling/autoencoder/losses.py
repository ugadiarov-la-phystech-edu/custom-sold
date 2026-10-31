from modeling.autoencoder.base import Autoencoder
import torch
import torch.nn.functional as F
from typing import Any, Dict


def reconstruction_loss(autoencoder: Autoencoder, images: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
    """The autoencoder's outputs for an image sequence, plus "reconstruction_loss" and the input "images".

    images: (batch_size, sequence_length, 3, height, width); actions: (batch_size, sequence_length, action_dim), where
    actions[:, t] led to images[:, t] (NaN at t = 0).
    """
    outputs = autoencoder(images, actions[:, 1:])
    loss = F.mse_loss(outputs["reconstructions"], images)
    return {**outputs, "reconstruction_loss": loss, "images": images}
