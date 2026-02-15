import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Optional

from .hp3o_trajectory_buffer import HP3OTrajectoryBuffer
from ..base.training.orchestrators.orchestrator import Orchestrator
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.callbacks.training_callback import TrainingCallback
from ..base.training.callbacks.logging.loggers.training_logger import TrainingLogger
from blockchain_mdps import BlockchainModel

class HP3OOrchestrator(Orchestrator):
    def __init__(self, algorithm: RLAlgorithm, loggers: Dict[str, TrainingLogger], callback: TrainingCallback,
                 blockchain_model: BlockchainModel, **kwargs):
        # Initialize components needed by create_replay_buffer() before calling super().__init__()
        self.device = kwargs.get('device', torch.device('cpu'))
        self.trajectory_buffer_size = kwargs.get('trajectory_buffer_size', 10)
        
        self.trajectory_buffer = HP3OTrajectoryBuffer(
            buffer_size_trajectories=self.trajectory_buffer_size,
            gamma=kwargs.get('gamma', 0.99),
            lam=kwargs.get('lam', 0.95),
            device=self.device
        )

        super().__init__(algorithm, loggers, callback, blockchain_model, **kwargs)
        
        self.hp3o_epochs = kwargs.get('hp3o_epochs', 10)
        self.hp3o_batch_trajectories = kwargs.get('hp3o_batch_trajectories', 4)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.value_coef = kwargs.get('value_coef', 0.5)
        self.entropy_coef = kwargs.get('entropy_coef', 0.01)
        self.hp3o_plus = kwargs.get('hp3o_plus', False)

        # Attributes expected by logging callbacks (TensorboardLoggingCallback etc.)
        self.number_of_training_agents = kwargs.get('number_of_training_agents', 1)
        self.number_of_evaluation_agents = kwargs.get('number_of_evaluation_agents', 1)
        self.train_episode_length = kwargs.get('train_episode_length', 1000)
        self.epoch_size = self.hp3o_batch_trajectories * self.train_episode_length
        

    def create_replay_buffer(self):
        # The logging callback expects a replay_buffer attribute.
        # We hook it to our trajectory_buffer.
        self.replay_buffer = self.trajectory_buffer
        return self.trajectory_buffer

    def run(self) -> None:
        self.run_training_epochs()

    def run_training_epochs(self) -> None:
        for epoch_idx in range(self.num_of_epochs):
            self.callback.before_training_epoch(epoch_idx)
            
            # Step 1: Gather a new trajectory
            # Use train_episode_length for trajectory size
            train_episode_length = self.creation_args.get('train_episode_length', 1000)
            exp = self.run_episode(epoch_idx, train_episode_length, evaluation=False)
            
            # HP3OAgent stores the trajectory
            trajectory = self.agent.get_full_trajectory()
            self.trajectory_buffer.append(trajectory)
            
            # Step 2: Optimize using the buffer
            if len(self.trajectory_buffer.trajectories) >= 1:
                self.optimize_hp3o()
            
            # Step 3: Evaluate
            self.run_episode(epoch_idx, self.evaluate_episode_length, evaluation=True)
            
            stop = self.callback.after_training_epoch(epoch_idx)
            if stop:
                break
            
            self.lr_scheduler.step()

    def gather_experience(self) -> bool:
        # Not used in this custom loop, but required by abstract
        return True

    def optimize_hp3o(self) -> None:
        self.approximator.train()
        
        # Sample batch of trajectories (including best)
        batch = self.trajectory_buffer.get_batch(self.hp3o_batch_trajectories, hp3o_plus=self.hp3o_plus)
        
        states = batch['states']
        actions = batch['actions']
        old_log_probs = batch['log_probs']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.hp3o_epochs):
            # Forward pass
            out = self.approximator(states)
            # out is [Batch, 1 + NumActions] because dim_out was [1, NumActions]
            values = out[:, 0]
            logits = out[:, 1:]
            
            # Mask illegal actions if necessary
            # For simplicity, we assume the logits from approximator are raw
            legal_actions = batch['legal_actions']
            logits = logits.masked_fill(~legal_actions, float('-inf'))
            
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO Loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values, returns)
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.approximator.parameters(), 0.5)
            self.optimizer.step()
            
        self.update()

    def update(self) -> None:
        self.update_agent()

        # Update the target approximator in the loss function if exists
        if self.loss_fn is not None:
            self.loss_fn.update()

        self.callback.after_training_update()

    def update_agent(self) -> None:
        # Sync weights from approximator to agent's local copy if any
        self.agent.update(self.approximator)
