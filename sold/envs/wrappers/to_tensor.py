import gym
import torch


class ToTensor(gym.Wrapper):
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict]:
        obs, reward, done, info = self.env.step(action.detach().cpu().numpy())
        return torch.from_numpy(obs), reward, done, info

    def reset(self) -> torch.Tensor:
        return torch.from_numpy(self.env.reset())
