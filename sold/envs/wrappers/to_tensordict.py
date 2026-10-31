import gymnasium as gym
import tensordict
import torch
from torchvision import transforms


def TransformObservation(obs: dict):
    ret = tensordict.TensorDict()
    for k, v in obs.items():
        if "image" in k:
            ret[k] = transforms.ToTensor()(v.copy())
        else:
            ret[k] = torch.Tensor(v.copy())
    return ret


class ToTensorDict(gym.Wrapper):
    def step(self, action: torch.Tensor) -> tuple[tensordict.TensorDict, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action.detach().cpu().numpy())
        return TransformObservation(obs).copy(), reward, terminated, truncated, info

    def reset(self) -> tuple[tensordict.TensorDict, dict]:
        obs, info = self.env.reset()
        return TransformObservation(obs), info
