import colorsys
import math
import matplotlib as mpl
import numpy as np
from PIL import ImageDraw
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms.functional import rgb_to_grayscale


def slot_color(slot_index: int, num_slots: int, saturation: float = 1.0, lightness: float = 0.5) -> torch.Tensor:
    num_slots = max(num_slots, 2)
    hue = (slot_index / (num_slots - 1)) * (1 - (1 / num_slots))
    return torch.Tensor(colorsys.hls_to_rgb(hue, lightness, saturation))


def make_grid(
    images: torch.Tensor, num_columns: int, padding: int = 2, pad_color: torch.Tensor | None = None
) -> torch.Tensor:
    if not torch.is_tensor(images):
        raise TypeError(f"Expected images as type tensor, got {type(images)}")

    if pad_color is None:
        pad_color = torch.Tensor([0.0, 0.0, 0.0])
    if pad_color.numel() != 3:
        raise ValueError("pad_color must have exactly 3 elements (RGB)")

    if images.ndim != 4:
        raise ValueError("Expected images of shape (num_images, num_channels, height, width)")
    num_images, num_channels, height, width = images.size()
    if num_channels != 3:
        raise ValueError("Expected images with 3 channels (RGB)")

    # Create background grid with pad_color.
    num_columns = min(num_columns, num_images)
    num_rows = int(math.ceil(float(num_images) / num_columns))
    height, width = int(height + padding), int(width + padding)
    grid = (
        pad_color.unsqueeze(1)
        .unsqueeze(2)
        .repeat(1, height * num_rows + padding, width * num_columns + padding)
        .to(images.device, images.dtype)
    )

    # Copy each image into its corresponding position in the grid.
    image_index = 0
    for row in range(num_rows):
        for col in range(num_columns):
            if image_index >= num_images:
                break
            grid.narrow(1, row * height + padding, height - padding).narrow(
                2, col * width + padding, width - padding
            ).copy_(images[image_index])
            image_index += 1
    return grid


def make_row(
    images: torch.Tensor,
    max_sequence_length: int | None = None,
    padding: int = 2,
    pad_color: torch.Tensor | None = None,
) -> torch.Tensor:
    sequence_length, _, _, _ = images.size()
    n_cols = min(sequence_length, max_sequence_length) if max_sequence_length is not None else sequence_length
    return make_grid(images[:n_cols], num_columns=n_cols, padding=padding, pad_color=pad_color)


def stack_rows(rows: list[torch.Tensor], height_spacing: int = 2) -> torch.Tensor:
    grid = []
    for row_index, row in enumerate(rows):
        grid.append(row.cpu())
        if row_index < len(rows) - 1:
            grid.append(torch.ones(3, height_spacing, row.size(2)))
    return torch.cat(grid, dim=1)


def stack_columns(
    images: torch.Tensor, num_context: int, pad_color: torch.Tensor | None = None, width_spacing: int = 2
) -> torch.Tensor:
    sequence_length = images.size(0)
    num_predictions = sequence_length - num_context
    context = make_grid(images[:num_context].cpu(), num_columns=num_context, pad_color=pad_color)
    prediction = make_grid(images[num_context:].cpu(), num_columns=num_predictions, pad_color=pad_color)
    return join_columns(context, prediction, width_spacing)


def join_columns(left: torch.Tensor, right: torch.Tensor, width_spacing: int = 2) -> torch.Tensor:
    """Place two images of the same height side by side, separated by a white column."""
    return torch.cat([left, torch.ones(3, left.size(1), width_spacing), right], dim=2)


def create_segmentation_overlay(
    images: torch.Tensor, masks: torch.Tensor, background_brightness: float = 0.4
) -> torch.Tensor:
    sequence_length, num_slots, _, width, height = masks.size()
    segmentations = background_brightness * rgb_to_grayscale(images, num_output_channels=3)
    for slot_index in range(num_slots):
        segmentations[:] += (
            (1 - background_brightness)
            * masks[:, slot_index].repeat(1, 3, 1, 1)
            * slot_color(slot_index, num_slots)
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(sequence_length, 1, width, height)
        )
    return segmentations


def draw_ticks(backgrounds: torch.Tensor, ticks, color):
    labeled_images = []
    for index, image in enumerate(backgrounds):
        img = torchvision.transforms.functional.to_pil_image(image)
        draw = ImageDraw.Draw(img)
        draw.text((0.5 * img.width, img.height), ticks[index], color, anchor="md")
        labeled_images.append(torchvision.transforms.functional.pil_to_tensor(img) / 255.0)

    return torch.stack(labeled_images)


def visualize_reward_prediction(images, reconstructions, rewards, predicted_rewards) -> torch.Tensor:
    images, reconstructions = images.cpu(), reconstructions.cpu()
    images = draw_reward(images, rewards.detach().cpu()) / 255
    reconstructions = draw_reward(reconstructions, predicted_rewards.detach().cpu()) / 255

    sequence_length, _, _, _ = images.size()
    true_row = make_grid(images, num_columns=sequence_length)
    model_row = make_grid(reconstructions, num_columns=sequence_length)
    return stack_rows([true_row, model_row])


def draw_reward(observation, reward, color=(255, 255, 255)):
    imgs = []
    for i, img in enumerate(observation):
        img = torchvision.transforms.functional.to_pil_image(img)
        draw = ImageDraw.Draw(img)
        draw.text((0.25 * img.width, 0.8 * img.height), f"{reward[i]:.3f}", color)
        imgs.append(torchvision.transforms.functional.pil_to_tensor(img))
    return torch.stack(imgs)


@torch.no_grad()
def get_attention_weights(model: nn.Module, slots: torch.Tensor) -> torch.Tensor:
    """Attention of the predictor's output token at the last time step on each slot, by attention rollout.

    Inspired by https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb

    Args:
        model (nn.Module): A predictor with a slot_aggregation_transformer (see modeling.sold.prediction).
        slots (torch.Tensor): Slot history of shape (1, sequence_length, num_slots, slot_dim).

    Returns:
        torch.Tensor: Weights of shape (sequence_length, num_slots), scaled to a maximum of 1.
    """
    batch_size, sequence_length, num_slots, slot_dim = slots.size()
    _, layer_weights = model.slot_aggregation_transformer.forward_with_attention(
        slots.detach(), start=sequence_length - 1
    )
    att_mat = torch.stack(layer_weights)

    # Account for residual connections: add the identity and re-normalize.
    aug_att_mat = att_mat + torch.eye(att_mat.size(2), device=slots.device)
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices.
    joint_attention = aug_att_mat[0]
    for layer_attention in aug_att_mat[1:]:
        joint_attention = torch.matmul(layer_attention, joint_attention)

    # Attention of the last output token, without the output and register tokens.
    output_attention = torch.chunk(joint_attention[0, -1, :], sequence_length, dim=-1)
    output_attention = torch.stack([frame_attention[:num_slots] for frame_attention in output_attention])
    return output_attention / output_attention.max()


def get_output_attention_images(attention_weights, rgbs, masks):
    cmap = mpl.colormaps["plasma"]

    attention_brightness_images = torch.sum(
        attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * rgbs * masks, dim=1
    )

    attention_colormap_images = torch.sum(attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * masks, dim=1)
    attention_colormap_images -= attention_colormap_images.min()
    attention_colormap_images /= attention_colormap_images.max()
    attention_colormap_images = torch.from_numpy(cmap(attention_colormap_images.cpu().numpy())).float()
    attention_colormap_images = attention_colormap_images[:, 0].permute(0, 3, 1, 2)[:, 0:3]
    return attention_brightness_images, attention_colormap_images


def visualize_output_attention(attention_weights, reconstructions, rgbs, masks):
    attention_brightness_images, attention_colormap_images = get_output_attention_images(attention_weights, rgbs, masks)
    sequence_length, _, _, _ = reconstructions.size()
    reconstructions_row = make_grid(reconstructions, num_columns=sequence_length).cpu()
    attention_brightness_row = make_grid(attention_brightness_images, num_columns=sequence_length).cpu()
    attention_colormap_row = make_grid(attention_colormap_images, num_columns=sequence_length).cpu()
    return stack_rows([reconstructions_row, attention_brightness_row, attention_colormap_row])


@torch.no_grad()
def visualize_reward_predictor_attention(
    images, reconstructions, rewards, predicted_rewards, attention_weights, predicted_rgbs, predicted_masks
) -> torch.Tensor:
    images, reconstructions = images.clone(), reconstructions.clone()  # Label copies, not the caller's tensors.
    images[-1:] = draw_reward(images[-1:], rewards[-1:].detach().cpu(), color=(0, 255, 0)) / 255
    reconstructions[-1:] = (
        draw_reward(reconstructions[-1:], predicted_rewards[-1:].detach().cpu(), color=(255, 0, 0)) / 255
    )
    sequence_length, _, _, _ = images.size()

    true_row = make_grid(images, num_columns=sequence_length)
    model_row = make_grid(reconstructions, num_columns=sequence_length)

    attention_brightness_images, attention_colormap_images = get_output_attention_images(
        attention_weights, predicted_rgbs, predicted_masks
    )
    attention_brightness_row = make_grid(attention_brightness_images, num_columns=sequence_length).cpu()
    attention_colormap_row = make_grid(attention_colormap_images, num_columns=sequence_length).cpu()
    return stack_rows([true_row, model_row, attention_brightness_row, attention_colormap_row])


# ------------------------------------------------------------------------------- SOLD figures
# Figures of a training batch's first sample, logged by SOLDModule after each evaluation. reconstruction: the
# autoencoder's outputs for the batch (incl. "images" and "slots", see modeling.autoencoder.losses).
_ALL_STEPS = slice(None)


def _first_sample(outputs: dict[str, torch.Tensor], steps: slice = _ALL_STEPS) -> dict[str, torch.Tensor]:
    return {k: v[0, steps] for k, v in outputs.items() if torch.is_tensor(v) and v.dim() > 1}


@torch.no_grad()
def dynamics_figure(
    autoencoder: nn.Module,
    reconstruction: dict[str, torch.Tensor],
    num_context: int,
    predicted_outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Context frames and reconstructions next to the dynamics model's predictions of the following frames.

    predicted_outputs: the decoded predictions of the horizon frames after the num_context context frames.
    """
    horizon = predicted_outputs["reconstructions"].size(1)
    x_ticks = np.array(["T=0"] + [str(t) for t in range(1, num_context + horizon)])
    context = _first_sample(reconstruction, slice(num_context)) | {"xticks": x_ticks[:num_context]}
    predicted_images = reconstruction["images"][0, num_context : num_context + horizon]
    prediction = _first_sample(predicted_outputs) | {"images": predicted_images, "xticks": x_ticks[num_context:]}
    return join_columns(autoencoder.visualize_reconstruction(context), autoencoder.visualize_reconstruction(prediction))


@torch.no_grad()
def reward_figures(
    reward_predictor: nn.Module,
    reconstruction: dict[str, torch.Tensor],
    rewards: torch.Tensor,
    predicted_rewards: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Predicted vs. true rewards and, for slot-based autoencoders, the reward predictor's attention."""
    images, reconstructions = reconstruction["images"][0], reconstruction["reconstructions"][0]
    figures = {
        "reward_prediction": visualize_reward_prediction(images, reconstructions, rewards[0], predicted_rewards[0])
    }
    if "masks" in reconstruction:
        attention_weights = get_attention_weights(reward_predictor, reconstruction["slots"][0:1])
        figures["reward_predictor_attention"] = visualize_reward_predictor_attention(
            images,
            reconstructions,
            rewards[0],
            predicted_rewards[0],
            attention_weights,
            reconstruction["rgbs"][0],
            reconstruction["masks"][0],
        )
    return figures


@torch.no_grad()
def imagination_figures(
    autoencoder: nn.Module,
    actor: nn.Module,
    imagined_slots: torch.Tensor,
    num_context: int,
    predicted_rewards: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """An imagined trajectory with predicted rewards and, for slot-based autoencoders, the actor's attention."""
    outputs = autoencoder.decode(imagined_slots[0:1])
    rewards = predicted_rewards[0].tolist()
    x_ticks = np.array([""] * num_context + [f"rew={rewards[0]:.2f}"] + [f"{r:.2f}" for r in rewards[1:]])
    context = _first_sample(outputs, slice(num_context)) | {"xticks": x_ticks[:num_context]}
    imagined = _first_sample(outputs, slice(num_context, None)) | {"xticks": x_ticks[num_context:]}
    figures = {
        "latent_imagination": join_columns(
            autoencoder.visualize_reconstruction(context), autoencoder.visualize_reconstruction(imagined)
        )
    }
    if "masks" in outputs:
        attention_weights = get_attention_weights(actor, imagined_slots[:1])
        figures["actor_attention"] = visualize_output_attention(
            attention_weights, outputs["reconstructions"][0], outputs["rgbs"][0], outputs["masks"][0]
        )
    return figures
