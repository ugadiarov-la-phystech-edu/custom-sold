from collections import defaultdict
from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass
import gym
import hydra
import numpy as np
from omegaconf import DictConfig
import os
from sold import rl
from sold.modeling.autoencoder.base import Autoencoder
from sold.modeling.autoencoder.losses import reconstruction_loss
from sold.modeling.distributions import Moments
from sold.utils.module import FreezeParameters
from sold.utils.training import OnlineModule, print_summary, set_seed
from sold.utils.visualizations import dynamics_figure, imagination_figures, reward_figures
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from typing import Any


@dataclass(frozen=True)
class OptimizerSettings:
    """Adam learning rate and gradient-norm clip of one model.

    warmup_updates > 0 raises the learning rate linearly from 0 over that many gradient updates after resuming from a
    checkpoint (e.g. for an optimizer whose state the checkpoint lacks).
    """

    learning_rate: float
    grad_clip: float
    warmup_updates: int = 0


OPTIMIZED_MODELS = ("dynamics", "reward", "actor", "critic", "autoencoder")


def optimization_settings(
    optimization: Mapping[str, Mapping[str, float]] | None, legacy_hparams: dict[str, Any]
) -> dict[str, dict[str, float]]:
    """The {learning_rate, grad_clip, warmup_updates} settings of each model in OPTIMIZED_MODELS.

    Old checkpoints store them as flat <model>_learning_rate and <model>_grad_clip hyperparameters, which are moved
    from legacy_hparams into the result (explicit optimization settings win).
    """
    grouped = {model: dict(settings) for model, settings in (optimization or {}).items()}
    for key in list(legacy_hparams):
        model, _, setting = key.partition("_")
        if model not in OPTIMIZED_MODELS:
            raise TypeError(f"Unexpected argument: {key}.")
        grouped.setdefault(model, {}).setdefault(setting, legacy_hparams.pop(key))
    return {model: asdict(OptimizerSettings(**grouped[model])) for model in OPTIMIZED_MODELS}


@dataclass
class Imagination:
    """Outputs of SOLDModule.imagine_ahead for horizon imagined steps."""

    lambda_returns: torch.Tensor  # (batch_size, horizon)
    predicted_values_targ: torch.Tensor  # Target critic's values, (batch_size, horizon).
    action_log_probs: torch.Tensor  # (batch_size, horizon)
    action_entropies: torch.Tensor  # (batch_size, horizon, 1)
    imagined_slots: torch.Tensor  # Context and imagined slots (detached), (batch_size, num_context + horizon, ...)
    num_context: int
    predicted_rewards: torch.Tensor  # (batch_size, horizon)


class SOLDModule(OnlineModule):
    GRADIENT_NORMS = ("reward_predictor", "actor", "critic")
    STAGES = ("autoencoder", "dynamics_loss", "reward_loss", "imagination", "actor_loss", "critic_loss")

    def __init__(
        self,
        autoencoder: Autoencoder,
        dynamics_predictor,
        actor,
        critic,
        reward_predictor,
        finetune_autoencoder: bool,
        num_context: int | tuple[int, int],
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
        profile: bool = True,
        keep_last_checkpoint_only: bool = False,
        optimization: Mapping[str, Mapping[str, float]] | None = None,  # Or flat legacy_hparams (old checkpoints).
        **legacy_hparams: Any,
    ) -> None:
        optimization = optimization_settings(optimization, legacy_hparams)
        # Every constructor argument but env, as Lightning's save_hyperparameters stored them in checkpoints.
        hparams = {
            name: value
            for name, value in locals().items()
            if name not in ("self", "env", "legacy_hparams", "__class__")
        }
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
            profile,
            keep_last_checkpoint_only,
        )
        self.hparams = hparams

        regression_infos = {
            "max_episode_steps": env.max_episode_steps,
            "num_slots": autoencoder.num_slots,
            "slot_dim": autoencoder.slot_dim,
        }
        self.autoencoder = autoencoder
        if not finetune_autoencoder:  # Its weights are never updated, so they collect no gradients.
            self.autoencoder.requires_grad_(False)
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

        self.optimizer_settings = {model: OptimizerSettings(**settings) for model, settings in optimization.items()}
        self.actor_entropy_loss_weight = actor_entropy_loss_weight
        self.actor_gradients = actor_gradients
        self.finetune_autoencoder = finetune_autoencoder
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

    def configure_optimizers(self) -> list[Optimizer]:
        optimizers = [
            self._adam("dynamics", self.dynamics_predictor),
            self._adam("reward", self.reward_predictor),
            self._adam("actor", self.actor),
            self._adam("critic", self.critic),
        ]
        if self.finetune_autoencoder:  # Returned here so that its state is saved in checkpoints.
            optimizers.append(self._adam("autoencoder", self.autoencoder))
        return optimizers

    def training_step(self, batch: dict[str, torch.Tensor], visualize: bool = False) -> dict[str, Any]:
        dynamics_optimizer, reward_optimizer, actor_optimizer, critic_optimizer, *autoencoder_optimizers = (
            self.optimizers
        )
        images, actions, rewards = batch["obs"] / 255.0, batch["action"], batch["reward"]

        # Without fine-tuning, the autoencoder forward pass needs no autograd graph.
        with self.measure_gpu("learn/autoencoder"), torch.set_grad_enabled(self.finetune_autoencoder):
            reconstruction = self._autoencoder_stage(images, actions)
        if self.finetune_autoencoder:
            self._optimize("autoencoder", autoencoder_optimizers[0], reconstruction["reconstruction_loss"])

        # Detach slots to prevent gradients from flowing back to the autoencoder model.
        slots = reconstruction["slots"].detach()
        outputs = dict(reconstruction)

        # Learn to predict dynamics in slot-space.
        with self.measure_gpu("learn/dynamics_loss"):
            outputs |= self.compute_dynamics_loss(images, slots, actions)
        self._optimize("dynamics", dynamics_optimizer, outputs["dynamics_loss"])

        # Learn to predict rewards from the slot representation.
        with self.measure_gpu("learn/reward_loss"):
            outputs |= self.compute_reward_loss(images, outputs["reconstructions"], slots, rewards)
        self._optimize("reward", reward_optimizer, outputs["reward_loss"])

        # Update the target critic network.
        with self.measure_gpu("learn/critic_target"):
            rl.returns.ema_update(self.critic_target, self.critic, self.critic_ema_decay)

        # Perform latent imagination to train the actor and critic.
        with self.measure_gpu("learn/imagination"):
            imagination = self.imagine_ahead(slots, actions)

        # Learn the actor.
        with self.measure_gpu("learn/actor_loss"):
            outputs |= self.compute_actor_loss(
                imagination.lambda_returns,
                imagination.predicted_values_targ,
                imagination.action_log_probs,
                imagination.action_entropies,
            )
        self._optimize("actor", actor_optimizer, outputs["actor_loss"])

        # Learn the critic.
        with self.measure_gpu("learn/critic_loss"):
            outputs |= self.compute_critic_loss(
                imagination.imagined_slots,
                imagination.num_context,
                imagination.lambda_returns,
                imagination.predicted_values_targ,
            )
        self._optimize("critic", critic_optimizer, outputs["critic_loss"])

        if visualize:
            with self.measure_gpu("learn/visualization"):
                self.log_visualizations(rewards, reconstruction, outputs, imagination)
        return outputs

    @torch.no_grad()
    def log_visualizations(
        self, rewards: torch.Tensor, reconstruction: dict[str, Any], outputs: dict[str, Any], imagination: Imagination
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

    def _autoencoder_stage(self, images: torch.Tensor, actions: torch.Tensor) -> dict[str, Any]:
        return reconstruction_loss(self.autoencoder, images, actions)

    def _adam(self, model_name: str, model: torch.nn.Module) -> torch.optim.Adam:
        learning_rate = self.optimizer_settings[model_name].learning_rate
        return torch.optim.Adam(model.parameters(), lr=learning_rate, fused=self.fused_optimizers or None)

    def _optimize(self, model_name: str, optimizer, loss: torch.Tensor) -> None:
        """One gradient step on loss, clipping the gradient norm to the model's grad_clip."""
        with self.measure_gpu(f"learn/{model_name}_update"):
            optimizer.zero_grad()
            loss.backward()
            grad_clip = self.optimizer_settings[model_name].grad_clip
            if grad_clip > 0:
                clip_grad_norm_([param for group in optimizer.param_groups for param in group["params"]], grad_clip)
            self._warm_up(model_name, optimizer)
            optimizer.step()

    def _warm_up(self, model_name: str, optimizer) -> None:
        """Scale the learning rate during the model's warmup after a resume (see OptimizerSettings)."""
        settings = self.optimizer_settings[model_name]
        if settings.warmup_updates == 0 or self.num_updates_since_restore is None:
            return
        scale = min(1.0, (self.num_updates_since_restore + 1) / settings.warmup_updates)
        self.learning_rates[model_name] = settings.learning_rate * scale
        for group in optimizer.param_groups:
            group["lr"] = self.learning_rates[model_name]

    def sample_num_context(self) -> int:
        return torch.randint(self.min_num_context, self.max_num_context + 1, (1,)).item()

    # Each training stage is a public method that switches module modes, freezes parameters and draws num_context,
    # around a _<stage>_stage method of tensors only, which compile_stages can replace by its compiled version.
    def compute_dynamics_loss(self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor) -> dict[str, Any]:
        self.dynamics_predictor.train()
        num_context = self.sample_num_context()
        return self._dynamics_loss_stage(images, slots, actions, num_context) | {"num_context": num_context}

    def _dynamics_loss_stage(
        self, images: torch.Tensor, slots: torch.Tensor, actions: torch.Tensor, num_context: int
    ) -> dict[str, Any]:
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
            "predicted_outputs": predicted_outputs,
        }

    def compute_reward_loss(
        self, images: torch.Tensor, reconstructions: torch.Tensor, slots: torch.Tensor, rewards: torch.Tensor
    ) -> dict[str, Any]:
        return self._reward_loss_stage(slots, rewards)

    def _reward_loss_stage(self, slots: torch.Tensor, rewards: torch.Tensor) -> dict[str, Any]:
        is_firsts = torch.isnan(rewards)  # We add NaN as a reward on the first time-step.
        predicted_rewards_dist = self.reward_predictor(slots.detach())
        log_probs = predicted_rewards_dist.log_prob(torch.nan_to_num(rewards).unsqueeze(2))
        # Masked means: boolean indexing syncs with the GPU.
        valid = (~is_firsts).to(log_probs.dtype)
        num_valid = valid.sum()

        predicted_rewards = predicted_rewards_dist.mean.squeeze(2).detach()
        squared_errors = (predicted_rewards - torch.nan_to_num(rewards)) ** 2
        # A metric: detached, converted to a number only when logged (no GPU sync).
        return {
            "reward_loss": -(log_probs.reshape(valid.shape) * valid).sum() / num_valid,
            "predicted_rewards": predicted_rewards,
            "reward_mse_loss": (squared_errors * valid).sum() / num_valid,
        }

    def imagine_ahead(self, slots: torch.Tensor, actions: torch.Tensor) -> Imagination:
        self.dynamics_predictor.eval()
        num_context = self.max_num_context if self.start_imagination_from_every else self.sample_num_context()
        # Only the actor (and through the imagined slots the dynamics model) gets gradients.
        with FreezeParameters([self.reward_predictor, self.critic]):
            outputs = self._imagination_stage(slots, actions, num_context)
        return Imagination(**outputs, num_context=num_context)

    def _imagination_stage(self, slots: torch.Tensor, actions: torch.Tensor, num_context: int) -> dict[str, Any]:
        action_log_probs, action_entropies = [], []
        if self.start_imagination_from_every:
            slots_context = (
                slots.unfold(dimension=1, size=self.max_num_context, step=1).flatten(end_dim=1).permute(0, 3, 1, 2)
            )
            actions_context = (
                actions.unfold(dimension=1, size=self.max_num_context, step=1)
                .flatten(end_dim=1)
                .permute(0, 2, 1)[:, 1:]
            )
        else:
            slots_context = slots[:, :num_context].detach()
            actions_context = actions[:, 1:num_context].detach()

        for _ in range(self.imagination_horizon):
            action_dist = self.actor(slots_context.detach(), start=slots_context.shape[1] - 1)
            selected_action = action_dist.rsample().squeeze(1)
            actions_context = torch.cat([actions_context, selected_action.unsqueeze(1)], dim=1)
            action_log_probs.append(action_dist.log_prob(selected_action.unsqueeze(1)))
            action_entropies.append(action_dist.entropy())

            predicted_slots = self.dynamics_predictor.predict_slots(
                slots_context, actions_context, steps=1, num_context=slots_context.shape[1]
            )
            slots_context = torch.cat([slots_context, predicted_slots], dim=1)

        predicted_rewards = self.reward_predictor(slots_context, start=num_context).mean.squeeze(-1)
        predicted_values = self.critic(slots_context, start=num_context).mean.squeeze(-1)
        slots_context = slots_context.detach()
        return {
            "lambda_returns": self.compute_lambda_returns(predicted_rewards, predicted_values),
            "predicted_values_targ": self.critic_target(slots_context[:, :-1], start=num_context - 1).mean.squeeze(-1),
            "action_log_probs": torch.stack(action_log_probs, dim=1).squeeze(2),
            "action_entropies": torch.stack(action_entropies, dim=1),
            "imagined_slots": slots_context,
            "predicted_rewards": predicted_rewards.detach(),
        }

    def compute_actor_loss(
        self,
        lambda_returns: torch.Tensor,
        predicted_values_targ: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_entropies: torch.Tensor,
    ) -> dict[str, Any]:
        return self._actor_loss_stage(lambda_returns, predicted_values_targ, action_log_probs, action_entropies)

    def _actor_loss_stage(
        self,
        lambda_returns: torch.Tensor,
        predicted_values_targ: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_entropies: torch.Tensor,
    ) -> dict[str, Any]:
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
        self,
        imagined_slots: torch.Tensor,
        num_context: int,
        lambda_returns: torch.Tensor,
        predicted_values_targ: torch.Tensor,
    ) -> dict[str, Any]:
        return self._critic_loss_stage(imagined_slots, num_context, lambda_returns, predicted_values_targ)

    def _critic_loss_stage(
        self,
        imagined_slots: torch.Tensor,
        num_context: int,
        lambda_returns: torch.Tensor,
        predicted_values_targ: torch.Tensor,
    ) -> dict[str, Any]:
        return rl.losses.critic_loss(
            self.critic(imagined_slots[:, :-1], start=num_context - 1),
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
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    sold = hydra.utils.instantiate(cfg.model)
    logger = hydra.utils.instantiate(cfg.logger.tensorboard)
    sold.fit(
        logger,
        device=cfg.device,
        checkpoint=os.path.abspath(cfg.checkpoint) if cfg.checkpoint else None,
        replay_buffer=os.path.abspath(cfg.replay_buffer) if cfg.replay_buffer else None,
        fused_optimizers=cfg.fused_optimizers,
        compile_mode=cfg.compile.mode,
        compile_stages=cfg.compile.stages,
    )

    if cfg.logger.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    train()
