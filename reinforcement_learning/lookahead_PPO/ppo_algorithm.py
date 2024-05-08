from abc import ABC

from .ppo_agent import PPOAgent
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.function_approximation.mlp_approximator import MLPApproximator


class PPOAlgorithm(RLAlgorithm):
    def create_actor_critic(self) -> MLPApproximator:
        # Assume the actor and the critic share the same network architecture
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [64, 64])
        return MLPApproximator(self.device, self.simulator.state_space_dim, self.simulator.action_space_dim,
                               hidden_layers_sizes)

    def create_agent(self) -> PPOAgent:
        actor_critic = self.create_actor_critic()
        clip_param = self.creation_args.get('clip_param', 0.2)
        ppo_epoch = self.creation_args.get('ppo_epoch', 4)
        num_mini_batch = self.creation_args.get('num_mini_batch', 32)
        value_loss_coef = self.creation_args.get('value_loss_coef', 0.5)
        entropy_coef = self.creation_args.get('entropy_coef', 0.01)
        lr = self.creation_args.get('lr', 3e-4)
        eps = self.creation_args.get('eps', 1e-5)
        max_grad_norm = self.creation_args.get('max_grad_norm', 0.5)
        return PPOAgent(actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef, lr, eps,
                        max_grad_norm)
