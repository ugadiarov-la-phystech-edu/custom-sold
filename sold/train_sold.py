from collections import defaultdict
import copy
from dataclasses import dataclass
import gym
import hydra
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT
from modeling.distributions import Moments
from modeling.autoencoder.base import Autoencoder
from modeling.autoencoder.losses import reconstruction_loss
import numpy as np
from omegaconf import DictConfig
import os
import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import Any, Dict, Tuple
import rl.losses
import rl.returns
from utils.instantiate import instantiate_trainer
from utils.module import FreezeParameters
from utils.training import set_seed, print_summary, OnlineModule
from utils.visualizations import dynamics_figure, imagination_figures, reward_figures


@dataclass
class Imagination:
    """Outputs of SOLDModule.imagine_ahead for horizon imagined steps."""

    lambda_returns: torch.Tensor  # (batch_size, horizon)
    predicted_values_targ: torch.Tensor  # Target critic's values, (batch_size, horizon).
    predicted_values_dist: D.Distribution  # Critic's predictions, mean of shape (batch_size, horizon, 1).
    action_log_probs: torch.Tensor  # (batch_size, horizon)
    action_entropies: torch.Tensor  # (batch_size, horizon, 1)
    imagined_slots: torch.Tensor  # Context and imagined slots (detached), (batch_size, num_context + horizon, ...)
    num_context: int
    predicted_rewards: torch.Tensor  # (batch_size, horizon)


class SOLDModule(OnlineModule):
    def __init__(
        self,
        autoencoder: Autoencoder,
        dynamics_predictor,
        actor,
        critic,
        reward_predictor,
        dynamics_learning_rate: float,
        dynamics_grad_clip: float,
        actor_learning_rate: float,
        actor_grad_clip: float,
        critic_learning_rate: float,
        critic_grad_clip: float,
        reward_learning_rate: float,
        reward_grad_clip: float,
        finetune_autoencoder: bool,
        autoencoder_learning_rate: float,
        autoencoder_grad_clip: float,
        num_context: int | Tuple[int, int],
        imagination_horizon: int,
        start_imagination_from_every: bool,
        actor_entropy_loss_weight: float,
        actor_gradients: str,
        return_lambda: float,
        discount_factor: float,
        critic_ema_decay: float,
        env: gym.Env,
        max_steps: int,
        num_seed: int,
        update_freq: int,
        num_updates: int,
        eval_freq: int,
        num_eval_episodes: int,
        batch_size: int,
        buffer_capacity: int,
        save_replay_buffer: bool,
        critic_regularization_weight: float = 0.1,
    ) -> None:
        self.min_num_context, self.max_num_context = (
            (num_context, num_context) if isinstance(num_context, int) else num_context
        )
        if self.min_num_context > self.max_num_context:
            raise ValueError("min_num_context must be less than or equal to max_num_context.")
        sequence_length = imagination_horizon + self.max_num_context

        super().__init__(
            env,
            max_steps,
            num_seed,
            update_freq,
            num_updates,
            eval_freq,
            num_eval_episodes,
            batch_size,
            sequence_length,
            buffer_capacity,
            save_replay_buffer,
        )
        self.automatic_optimization = False
        self.save_hyperparameters(logger=False, ignore=["env"])

        regression_infos = {
            "max_episode_steps": env.max_episode_steps,
            "num_slots": autoencoder.num_slots,
            "slot_dim": autoencoder.slot_dim,
        }
        self.autoencoder = autoencoder
        self.actor = actor(
            **regression_infos,
            output_dim=env.action_space.shape[0],
            lower_bound=env.action_space.low,
            upper_bound=env.action_space.high,
        )
        self.critic = critic(**regression_infos)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.requires_grad_(False)  # Updated only by EMA, never by gradients.
        self.reward_predictor = reward_predictor(**regression_infos)
        self.dynamics_predictor = dynamics_predictor(
            num_slots=autoencoder.num_slots,
            slot_dim=autoencoder.slot_dim,
            sequence_length=imagination_horizon,
            action_dim=env.action_space.shape[0],
            input_buffer_size=sequence_length,
        )

        self.dynamics_learning_rate = dynamics_learning_rate
        self.dynamics_grad_clip = dynamics_grad_clip
        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_clip = actor_grad_clip
        self.actor_entropy_loss_weight = actor_entropy_loss_weight
        self.actor_gradients = actor_gradients
        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_clip = critic_grad_clip
        self.reward_learning_rate = reward_learning_rate
        self.reward_grad_clip = reward_grad_clip

        self.finetune_autoencoder = finetune_autoencoder
        self.autoencoder_grad_clip = autoencoder_grad_clip
        self.autoencoder_learning_rate = autoencoder_learning_rate
        self.imagination_horizon = imagination_horizon
        self.start_imagination_from_every = start_imagination_from_every
        self.return_lambda = return_lambda
        self.discount_factor = discount_factor
        self.critic_ema_decay = critic_ema_decay
        self.critic_regularization_weight = critic_regularization_weight

        self.return_moments = Moments()
        self.register_buffer("discounts", torch.full((1, self.imagination_horizon), self.discount_factor))
        self.discounts = torch.cumprod(self.discounts, dim=1) / self.discount_factor

        self.current_losses = defaultdict(list)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizers = [
            torch.optim.Adam(self.dynamics_predictor.parameters(), lr=self.dynamics_learning_rate),
            torch.optim.Adam(self.reward_predictor.parameters(), lr=self.reward_learning_rate),
            torch.optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate),
            torch.optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate),
        ]
        if self.finetune_autoencoder:  # Returned here so that its state is saved in checkpoints.
            optimizers.append(torch.optim.Adam(self.autoencoder.parameters(), lr=self.autoencoder_learning_rate))
        return optimizers

    def training_step(self, batch, batch_index: int) -> STEP_OUTPUT:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer, *autoencoder_optimizers = (
            self.optimizers()
        )
        images, actions, rewards = (
            batch["obs"].squeeze(0) / 255.0,
            batch["action"].squeeze(0),
            batch["reward"].squeeze(0),
        )

        # Without fine-tuning, the autoencoder forward pass needs no autograd graph.
        with torch.set_grad_enabled(self.finetune_autoencoder):
            reconstruction = reconstruction_loss(self.autoencoder, images, actions)
        if self.finetune_autoencoder:
            self._optimize(autoencoder_optimizers[0], reconstruction["reconstruction_loss"], self.autoencoder_grad_clip)

        # Detach slots to prevent gradients from flowing back to the autoencoder model.
        slots = reconstruction["slots"].detach()
        outputs = dict(reconstruction)

        # Learn to predict dynamics in slot-space.
        outputs |= self.compute_dynamics_loss(images, slots, actions)
        self._optimize(dynamics_optimizer, outputs["dynamics_loss"], self.dynamics_grad_clip)

        # Learn to predict rewards from the slot representation.
        outputs |= self.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
        self._optimize(reward_optimizer, outputs["reward_loss"], self.reward_grad_clip)

        # Update the target critic network.
        rl.returns.ema_update(self.critic_target, self.critic, self.critic_ema_decay)

        # Perform latent imagination to train the actor and critic.
        imagination = self.imagine_ahead(slots, actions)

        # Learn the actor.
        outputs |= self.compute_actor_loss(
            imagination.lambda_returns,
            imagination.predicted_values_targ,
            imagination.action_log_probs,
            imagination.action_entropies,
        )
        self._optimize(actor_optimizer, outputs["actor_loss"], self.actor_grad_clip)

        # Learn the critic.
        outputs |= self.compute_critic_loss(
            imagination.predicted_values_dist, imagination.lambda_returns, imagination.predicted_values_targ
        )
        self._optimize(critic_optimizer, outputs["critic_loss"], self.critic_grad_clip)

        if self.after_eval:
            self.log_visualizations(rewards, reconstruction, outputs, imagination)

        # Log all losses.
        for key, value in outputs.items():
            if key.endswith("_loss"):
                self.log("train/" + key, value)
        self.log_gradients(model_names=("reward_predictor", "actor", "critic"))
        return outputs

    @torch.no_grad()
    def log_visualizations(
        self, rewards: torch.Tensor, reconstruction: Dict[str, Any], outputs: Dict[str, Any], imagination: Imagination
    ) -> None:
        """Log figures of the first sample of a training batch (after each evaluation).

        reconstruction: the autoencoder's outputs; outputs: those of compute_dynamics_loss and compute_reward_loss.
        Attention maps show the reward predictor and actor after this step's update.
        """
        self.log_image(
            "autoencoder_reconstruction",
            self.autoencoder.visualize_reconstruction({k: v[0] for k, v in reconstruction.items() if v.dim() > 0}),
        )
        self.log_image(
            "dynamics_prediction",
            dynamics_figure(self.autoencoder, reconstruction, outputs["num_context"], outputs["predicted_outputs"]),
        )
        figures = reward_figures(self.reward_predictor, reconstruction, rewards, outputs["predicted_rewards"])
        figures |= imagination_figures(
            self.autoencoder,
            self.actor,
            imagination.imagined_slots,
            imagination.num_context,
            imagination.predicted_rewards,
        )
        for name, image in figures.items():
            self.log_image(name, image)

    def _optimize(self, optimizer, loss: torch.Tensor, grad_clip: float) -> None:
        """One gradient step on loss, with the gradient norm clipped to grad_clip."""
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=grad_clip, gradient_clip_algorithm="norm")
        optimizer.step()

    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> Dict[str, Any]:
        self.dynamics_predictor.train()
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()

        future_slots = self.dynamics_predictor.predict_slots(
            slots, actions[:, 1:].clone().detach(), steps=self.imagination_horizon, num_context=num_context
        )
        future_outputs = self.autoencoder.decode(future_slots)
        slot_loss = F.mse_loss(future_slots, slots[:, num_context : num_context + self.imagination_horizon])
        image_loss = F.mse_loss(
            future_outputs["reconstructions"], images[:, num_context : num_context + self.imagination_horizon]
        )

        # For figures: the first sample's decoded predictions (the batch is freed after the update).
        predicted_outputs = {k: v[:1].detach().clone() for k, v in future_outputs.items()}
        return {
            "slot_loss": slot_loss,
            "image_loss": image_loss,
            "dynamics_loss": slot_loss + image_loss,
            "num_context": num_context,
            "predicted_outputs": predicted_outputs,
        }

    def compute_reward_loss(
        self, images: torch.Tensor, reconstructions: torch.Tensor, slots: torch.Tensor, rewards: torch.Tensor
    ) -> Dict[str, Any]:
        is_firsts = torch.isnan(rewards)  # We add NaN as a reward on the first time-step.
        predicted_rewards_dist = self.reward_predictor(slots.detach())
        log_probs = predicted_rewards_dist.log_prob(torch.nan_to_num(rewards).unsqueeze(2))
        masked_log_probs = log_probs[~is_firsts]

        predicted_rewards = predicted_rewards_dist.mean.squeeze(2).detach()
        # A metric: detached, converted to a number only when logged (no GPU sync).
        return {
            "reward_loss": -masked_log_probs.mean(),
            "predicted_rewards": predicted_rewards,
            "reward_mse_loss": F.mse_loss(predicted_rewards[~is_firsts], rewards[~is_firsts]),
        }

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> Imagination:
        self.dynamics_predictor.eval()
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        action_log_probs, action_entropies = [], []

        if self.start_imagination_from_every:
            num_context = self.max_num_context
            slots_context = (
                slots.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 3, 1, 2)
            )
            actions_context = (
                actions.unfold(dimension=1, size=self.max_num_context, step=1)
                .flatten(end_dim=1)
                .permute(0, 2, 1)[:, 1:]
            )
        else:
            num_context = torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()
            slots_context = slots[:, :num_context].detach()
            actions_context = actions[:, 1:num_context].detach()

        # Actor update
        # Freeze models except action model and imagine next states
        with FreezeParameters([self.reward_predictor, self.critic]):
            for t in range(self.imagination_horizon):
                action_dist = self.actor(slots_context.detach(), start=slots_context.shape[1] - 1)
                selected_action = action_dist.rsample().squeeze(1)
                actions_context = torch.cat([actions_context, selected_action.unsqueeze(1)], dim=1)
                action_log_probs.append(action_dist.log_prob(selected_action.unsqueeze(1)))
                action_entropies.append(action_dist.entropy())

                predicted_slots = self.dynamics_predictor.predict_slots(
                    slots_context, actions_context, steps=1, num_context=slots_context.shape[1]
                )
                slots_context = torch.cat([slots_context, predicted_slots], dim=1)

        with FreezeParameters([self.reward_predictor, self.critic]):
            predicted_rewards = self.reward_predictor(slots_context, start=num_context).mean.squeeze(-1)
            predicted_values = self.critic(slots_context, start=num_context).mean.squeeze(-1)

        lambda_returns = self.compute_lambda_returns(predicted_rewards, predicted_values)

        action_log_probs = torch.stack(action_log_probs, dim=1).squeeze(2)
        action_entropies = torch.stack(action_entropies, dim=1)

        # Value update
        slots_context = slots_context.detach()
        # Predict imagined values
        predicted_values_targ = self.critic_target(slots_context[:, :-1], start=num_context - 1).mean.squeeze(-1)
        predicted_values_dist = self.critic(slots_context[:, :-1], start=num_context - 1)

        return Imagination(
            lambda_returns,
            predicted_values_targ,
            predicted_values_dist,
            action_log_probs,
            action_entropies,
            slots_context,
            num_context,
            predicted_rewards.detach(),
        )

    def compute_actor_loss(
        self,
        lambda_returns: torch.Tensor,
        predicted_values_targ: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_entropies: torch.Tensor,
    ) -> Dict[str, Any]:
        offset, invscale = self.return_moments(lambda_returns[:, :-1])  # Updates the running return percentiles.
        return rl.losses.actor_loss(
            lambda_returns,
            predicted_values_targ,
            action_log_probs,
            action_entropies,
            self.discounts,
            offset,
            invscale,
            self.actor_gradients,
            self.actor_entropy_loss_weight,
        )

    def compute_critic_loss(
        self, predicted_values_dist: D.Distribution, lambda_returns: torch.Tensor, predicted_values_targ: torch.Tensor
    ) -> Dict[str, Any]:
        return rl.losses.critic_loss(
            predicted_values_dist,
            lambda_returns,
            predicted_values_targ,
            self.discounts,
            self.critic_regularization_weight,
        )

    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return rl.returns.lambda_returns(rewards, values, self.discount_factor, self.return_lambda)

    def select_action(self, observation: torch.Tensor, is_first: bool = False, mode: str = "train") -> torch.Tensor:
        observation = observation.unsqueeze(0) / 255.0  # Expand batch dimension (1, 3, height, width).

        # Encode image into slots and append to context.
        last_slots = None if is_first else self._slot_history[:, -1]
        slots = self.autoencoder.encode(
            observation.unsqueeze(1), self.last_action.unsqueeze(0).unsqueeze(1), prior_slots=last_slots
        )  # Expand sequence (and batch) dimension.
        self._slot_history = slots if is_first else torch.cat([self._slot_history, slots], dim=1)

        if mode == "random":
            selected_action = torch.from_numpy(self.env.action_space.sample().astype(np.float32))
        else:
            action_dist = self.actor(self._slot_history, start=self._slot_history.shape[1] - 1)
            if mode == "train":
                selected_action = action_dist.sample().squeeze()
            elif mode == "eval":
                selected_action = action_dist.mode.squeeze()
            else:
                raise ValueError(f"Invalid mode: {mode}")

        # Clamp every dimension to its own bounds (the actor keeps them as buffers on the module's device).
        selected_action = selected_action.to(self.actor.lower_bound.device)
        return torch.clamp(selected_action, self.actor.lower_bound, self.actor.upper_bound).detach()


@hydra.main(config_path="../configs", config_name="train_sold", version_base=None)
def train(cfg: DictConfig):
    if cfg.logger.log_to_wandb:
        import wandb

        wandb.init(project="sold", name=cfg.experiment, config=dict(cfg), sync_tensorboard=True)
    print_summary(cfg)
    set_seed(cfg.seed)
    sold = hydra.utils.instantiate(cfg.model)
    trainer = instantiate_trainer(cfg)
    trainer.fit(sold, ckpt_path=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None)

    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    train()
