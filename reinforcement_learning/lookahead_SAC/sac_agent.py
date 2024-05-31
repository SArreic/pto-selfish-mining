from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from reinforcement_learning import MDPBlockchainSimulator
from reinforcement_learning.base.experience_acquisition.agents.ac_agent import ACAgent
from reinforcement_learning.base.function_approximation.approximator import Approximator


class SACAgent(ACAgent):
    def __init__(self, actor_approximator: Approximator, critic_approximator: Approximator,
                 simulator: MDPBlockchainSimulator, state_dim, action_dim, hidden_size, action_space):
        super().__init__(actor_approximator, critic_approximator, simulator)
        self.critic_approximator = critic_approximator
        self.action_space = action_space
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

        # Critic Networks
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Value Network
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)
        self.value_net_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)

        # Entropy coefficient
        self.target_entropy = -action_dim  # Target entropy
        self.log_alpha = torch.tensor([0.0], requires_grad=True)  # Initialize log_alpha
        self.alpha = self.log_alpha.exp()

        # Optimizer for alpha
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        # Target Value Network
        self.target_value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_value_net.load_state_dict(self.value_net.state_dict())

    def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
        with torch.no_grad():
            value = self.value_net(state)
        return value

    def plan_action(self, explore: bool = True) -> Tuple[int, torch.Tensor]:
        state = torch.Tensor
        state = torch.FloatTensor(state).unsqueeze(0)

        if explore:
            action_probabilities = torch.softmax(self.actor(state), dim=-1)
            action = action_probabilities.multinomial(num_samples=1)
        else:
            with torch.no_grad():
                q_values = self.critic1(torch.cat([state, self.actor(state)], 1))
            action = torch.argmax(q_values, dim=1)
            action_probabilities = torch.softmax(self.actor(state), dim=-1)

        return action.item(), action_probabilities

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probabilities = torch.softmax(self.actor(state), dim=-1)
        action = action_probabilities.multinomial(num_samples=1)
        return action.item()

    def update_parameters(self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
        # Convert batch data to tensors
        states = torch.FloatTensor(batch_states)
        actions = torch.LongTensor(batch_actions).unsqueeze(-1)
        rewards = torch.FloatTensor(batch_rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(batch_next_states)
        dones = torch.FloatTensor(batch_dones).unsqueeze(-1)

        # Compute the target for the Q networks
        with torch.no_grad():
            next_state_values = self.target_value_net(next_states).detach()
            q_target = rewards + (1 - dones) * 0.99 * next_state_values

        # Update Critic Networks
        q1_pred = self.critic1(torch.cat([states, actions], 1))
        q2_pred = self.critic2(torch.cat([states, actions], 1))
        critic1_loss = nn.MSELoss()(q1_pred, q_target)
        critic2_loss = nn.MSELoss()(q2_pred, q_target)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Value Network
        with torch.no_grad():
            # Sample action from the policy
            new_actions = self.actor(states)
            new_actions_probabilities = torch.softmax(new_actions, dim=-1)
            new_actions = new_actions_probabilities.multinomial(num_samples=1)

            # Compute the min Q value from the two critics
            min_q = torch.min(
                self.critic1(torch.cat([states, new_actions], 1)),
                self.critic2(torch.cat([states, new_actions], 1))
            ).detach()

            # Compute the log probabilities of the sampled actions
            new_actions_log_probs = torch.log(new_actions_probabilities + 1e-10).gather(1, new_actions)

        # The value target should be computed using the sampled actions' log probabilities
        value_pred = self.value_net(states)
        value_target = min_q - self.alpha * new_actions_log_probs
        value_loss = nn.MSELoss()(value_pred, value_target)

        self.value_net_optimizer.zero_grad()
        value_loss.backward()
        self.value_net_optimizer.step()

        # Update Target Value Network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * 0.995 + param.data * 0.005)

        # Update Policy Network (Actor)
        new_actions_probabilities = torch.softmax(self.actor(states), dim=-1)
        new_actions = new_actions_probabilities.multinomial(num_samples=1)
        new_actions_log_probs = torch.log(new_actions_probabilities + 1e-10).gather(1, new_actions)

        actor_loss = -(min_q - self.alpha * new_actions_log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Alpha:
        # Compute the entropy of the action probabilities
        action_probabilities = torch.softmax(self.actor(states), dim=-1)
        action_log_probs = torch.log(action_probabilities + 1e-10)
        action_entropy = -torch.sum(action_probabilities * action_log_probs, dim=1, keepdim=True)

        alpha_loss = -(self.log_alpha * (action_entropy + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        # Return losses for logging purposes
        return critic1_loss.item(), critic2_loss.item(), value_loss.item(), actor_loss.item()

# Example usage:
# agent = SACAgent(state_dim=state_space, action_dim=action_space, hidden_size=256, action_space=env.action_space)
# critic1_loss, critic2_loss, value_loss, actor_loss = agent.update_parameters(...)
