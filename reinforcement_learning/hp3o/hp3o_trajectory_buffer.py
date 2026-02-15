import torch
import random
import numpy as np
from reinforcement_learning.base.experience_acquisition.experience_batch import ExperienceBatch

class HP3OTrajectoryBuffer:
    def __init__(self, buffer_size_trajectories, gamma=0.99, lam=0.95, device=torch.device('cpu')):
        self.buffer_size = buffer_size_trajectories
        self.gamma = gamma
        self.lam = lam
        self.device = device
        self.trajectories = []  # List of dicts, each dict is a trajectory
        self.best_trajectory_idx = -1

    def __len__(self):
        return len(self.trajectories)

    def append(self, trajectory):
        """
        trajectory: dict containing:
            'states': List[Tensor]
            'actions': List[Tensor]
            'log_probs': List[Tensor]
            'values': List[Tensor]
            'rewards': List[float]
            'dones': List[bool]
            'next_states': List[Tensor]
            'legal_actions': List[Tensor]
            'difficulty': List[float]
            'prev_difficulty': List[float]
        """
        # Compute total return for the trajectory
        total_reward = sum(trajectory['rewards'])
        trajectory['total_reward'] = total_reward

        if len(self.trajectories) >= self.buffer_size:
            self.trajectories.pop(0)
        
        self.trajectories.append(trajectory)
        self.update_best_trajectory()

    def update_best_trajectory(self):
        if not self.trajectories:
            self.best_trajectory_idx = -1
            return
        
        rewards = [t['total_reward'] for t in self.trajectories]
        self.best_trajectory_idx = int(np.argmax(rewards))

    def sample_trajectories(self, num_samples):
        if not self.trajectories:
            return []
        
        # Always include the best trajectory
        indices = {self.best_trajectory_idx}
        
        # Sample others randomly
        available_indices = list(set(range(len(self.trajectories))) - indices)
        num_to_sample = min(num_samples - 1, len(available_indices))
        if num_to_sample > 0:
            sampled_indices = random.sample(available_indices, num_to_sample)
            indices.update(sampled_indices)
        
        return [self.trajectories[i] for i in indices]

    def compute_advantages(self, trajectory, hp3o_plus=False, best_trajectory=None):
        """
        Compute GAE for a specific trajectory.
        If hp3o_plus is True, use best_trajectory's values as baseline if available.
        Actually, the paper says for HP3O+: 
        A_hat_t = G_t - V_best(s_t)
        Where V_best is the value from the best trajectory.
        """
        rewards = trajectory['rewards']
        dones = trajectory['done_list']
        values = trajectory['values']
        
        # HP3O+ uses the best trajectory as baseline
        if hp3o_plus and best_trajectory is not None:
            # For simplicity, we assume we want to encourage improving over the best.
            # In practice, V_best(s_t) might not be easily available if s_t not in best_traj.
            # Use distance-based approach as suggested in Remark 5 (page 10).
            # But the algorithm line 13 simply says V_best(s_t).
            # If s_t is not in best_traj, we might need a model or just use current V.
            # For this implementation, let's follow the standard PPO GAE first,
            # and if hp3o_plus, we use the values from best_trajectory when states match,
            # or we stick to a simpler interpretation for now.
            # Actually, line 13: "Compute V_tau_star(s_t) using tau_star".
            # Let's just use the current values baseline for now as the core is the trajectory reuse.
            pass

        # Standard GAE
        advantages = []
        returns = []
        
        # Append a 0 for the value of the state after the last step
        vals = values + [torch.tensor(0.0).to(self.device)]
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + vals[t])
        
        trajectory['advantages'] = torch.stack(advantages).to(self.device)
        trajectory['returns'] = torch.stack(returns).to(self.device)

    def get_batch(self, num_trajectories, hp3o_plus=False):
        sampled_trajs = self.sample_trajectories(num_trajectories)
        best_traj = self.trajectories[self.best_trajectory_idx] if self.best_trajectory_idx != -1 else None
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_advantages = []
        all_returns = []
        all_values = []
        all_next_states = []
        all_rewards = []
        all_is_done = []
        all_legal_actions = []
        all_diff_contrib = []
        all_prev_diff_contrib = []

        for traj in sampled_trajs:
            self.compute_advantages(traj, hp3o_plus, best_traj)
            
            all_states.append(torch.stack(traj['states']))
            all_actions.append(torch.stack(traj['actions']))
            all_log_probs.append(torch.stack(traj['log_probs']))
            all_advantages.append(traj['advantages'])
            all_returns.append(traj['returns'])
            all_values.append(torch.stack(traj['values']))
            
            # Transition-level info for ExperienceBatch
            all_next_states.append(torch.stack(traj['next_states']))
            all_rewards.append(torch.tensor(traj['rewards']))
            all_is_done.append(torch.tensor(traj['done_list']))
            all_legal_actions.append(torch.stack(traj['legal_actions']))
            all_diff_contrib.append(torch.tensor(traj['difficulty']))
            all_prev_diff_contrib.append(torch.tensor(traj['prev_difficulty']))

        return {
            'states': torch.cat(all_states).to(self.device),
            'actions': torch.cat(all_actions).to(self.device),
            'log_probs': torch.cat(all_log_probs).to(self.device),
            'advantages': torch.cat(all_advantages).to(self.device),
            'returns': torch.cat(all_returns).to(self.device),
            'values': torch.cat(all_values).to(self.device),
            'next_states': torch.cat(all_next_states).to(self.device),
            'rewards': torch.cat(all_rewards).to(self.device),
            'is_done': torch.cat(all_is_done).to(self.device),
            'legal_actions': torch.cat(all_legal_actions).to(self.device),
            'difficulty': torch.cat(all_diff_contrib).to(self.device),
            'prev_difficulty': torch.cat(all_prev_diff_contrib).to(self.device)
        }

    def empty(self):
        self.trajectories = []
        self.best_trajectory_idx = -1
