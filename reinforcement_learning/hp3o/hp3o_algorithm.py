import torch
import torch.nn as nn
from typing import List

from .hp3o_agent import HP3OAgent
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.function_approximation.mlp_approximator import MLPApproximator

class HP3OAlgorithm(RLAlgorithm):
    def create_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('hidden_layers_sizes', [256, 256])
        dropout = self.creation_args.get('dropout', 0)
        
        # PPO/HP3O needs two heads: 
        # 1. State value (1)
        # 2. Action logits (num_actions)
        dim_out = [1, self.simulator.num_of_actions]
        
        return MLPApproximator(self.device, self.simulator.state_space_dim, dim_out,
                               hidden_layers_sizes, dropout, bias_in_last_layer=True)

    def create_agent(self) -> HP3OAgent:
        # Create a separate approximator instance for the agent's local copy
        # this matches the pattern in other algorithms like DQN
        agent_approximator = self.create_approximator()
        return HP3OAgent(agent_approximator, self.simulator)

    class DummyLoss(nn.Module):
        def update(self):
            pass

    def create_loss_fn(self):
        # We handle optimization in optimize_hp3o, but provide a dummy 
        # object to satisfy the checkpoint/logging system.
        return self.DummyLoss()

    def create_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.approximator.parameters(),
            lr=self.creation_args.get('learning_rate', 2e-4),
            weight_decay=self.creation_args.get('weight_decay', 0)
        )
