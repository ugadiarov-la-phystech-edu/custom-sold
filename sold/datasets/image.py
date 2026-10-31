"""Image datasets that provide training data for SAVi.

SAVi is trained on sequences of images (and optionally actions) to learn object-centric decompositions of the
visual scene. We provide datasets to load episodes stored in PNG or NPZ format, where the latter is especially useful
when file count limits apply, as common in HPC environments.
"""

import glob
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Any


class ImageDataset(Dataset):
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int) -> None:
        super().__init__()
        self.path = path
        self.data_dir = data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.split_path = os.path.join(self.path, self.data_dir, split)

        self.episode_dir_names = []
        for episode_dir in os.listdir(self.split_path):
            try:
                self.episode_dir_names.append(int(episode_dir))
            except ValueError:
                continue
        self.episode_dir_names.sort()

    @property
    def dataset_infos(self) -> dict[str, Any]:
        if not hasattr(self, "image_size") or not hasattr(self, "action_dim"):
            image_sequence, actions = self[0]
            self.image_size = list(image_sequence[0].shape[1:])
            self.action_dim = actions[0].shape[0]
        return {"image_size": self.image_size, "action_dim": self.action_dim}

    def __len__(self) -> int:
        return len(self.episode_dir_names)

    def _load_npz_data(self, index: int) -> list[np.array]:
        episode_dir = os.path.join(self.split_path, str(self.episode_dir_names[index]))
        episode_data = np.load(os.path.join(episode_dir, "episode.npz"))
        return [episode_data.get(key, None) for key in ["images", "actions", "rewards"]]

    def _random_start(self, episode_length: int) -> int:
        # The upper bound is exclusive, so +1 makes the last window (and episodes of exactly sequence_length) usable.
        return np.random.randint(0, episode_length - self.sequence_length + 1) if self.split == "train" else 0

    @staticmethod
    def _actions_leading_to_frames(actions: np.ndarray) -> np.ndarray:
        """Align actions with SOLD's replay buffer: entry t is the action that led to frame t (NaN for frame 0).

        Episode files store actions[t] = action taken after frame t, one fewer than there are frames.
        """
        return np.concatenate([np.full_like(actions[:1], np.nan), actions], axis=0)


class PNGDataset(ImageDataset):
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int) -> None:
        super().__init__(path, data_dir, split, sequence_length)
        self.episode_png_paths = []
        for dir_name in self.episode_dir_names:
            episode_dir = os.path.join(self.split_path, str(dir_name))
            png_paths = list(glob.glob(os.path.join(episode_dir, "*.png")))
            png_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.episode_png_paths.append(png_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        _, actions, rewards = self._load_npz_data(index)
        actions = self._actions_leading_to_frames(actions)
        png_paths = self.episode_png_paths[index]
        start_index = self._random_start(len(png_paths))

        image_sequence = []
        for image_index in range(start_index, start_index + self.sequence_length):
            image = Image.open(png_paths[image_index])
            image = transforms.ToTensor()(image)[:3]
            image_sequence.append(image)
        image_sequence = torch.stack(image_sequence, dim=0)
        action_sequence = torch.from_numpy(actions[start_index : start_index + self.sequence_length])
        return image_sequence, action_sequence


class NPZDataset(ImageDataset):
    def __init__(self, path: str, data_dir: str, split: str, sequence_length: int) -> None:
        super().__init__(path, data_dir, split, sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        images, actions, rewards = self._load_npz_data(index)
        actions = self._actions_leading_to_frames(actions)
        start_index = self._random_start(len(images))

        image_sequence = images[start_index : start_index + self.sequence_length]
        image_sequence = (torch.from_numpy(image_sequence) / 255.0).permute(0, 3, 1, 2)  # (sequence_length, 3, H, W)
        action_sequence = torch.from_numpy(actions[start_index : start_index + self.sequence_length])
        return image_sequence, action_sequence


def load_image_dataset(path: str, data_dir: str, split: str, sequence_length: int, **kwargs) -> ImageDataset:
    first_episode_dir = os.path.join(path, data_dir, split, "0")
    if any(f.endswith(".png") for f in os.listdir(first_episode_dir)):
        return PNGDataset(path, data_dir, split, sequence_length)
    else:
        return NPZDataset(path, data_dir, split, sequence_length)
