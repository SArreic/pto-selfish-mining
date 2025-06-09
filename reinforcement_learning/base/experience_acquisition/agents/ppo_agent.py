# === File: base/experience_acquisition/agents/ppo_agent.py ===

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from reinforcement_learning.base.experience_acquisition.experience import Experience


class PPOAgent:
    def __init__(self, policy_net, value_net, buffer, simulator, optimizer,
                 ppo_epochs=4, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.current_state = None
        self.policy_net = policy_net
        self.value_net = value_net
        self.buffer = buffer
        self.simulator = simulator
        self.optimizer = optimizer

        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def plan_action(self, state, explore=True):
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.value_net(state)

        self.buffer.store(state, action, log_prob, value)
        return action.item()

    def step(self):
        action, _ = self.plan_action(self.simulator.get_current_state())
        exp = self.simulator.step(action)
        legal_actions = self.simulator.get_state_legal_actions_tensor(exp.next_state)

        experience = Experience(
            prev_state=self.simulator.get_current_state(),
            action=action,
            next_state=exp.next_state,
            reward=exp.reward,
            difficulty_contribution=exp.difficulty_contribution,
            prev_difficulty_contribution=exp.prev_difficulty_contribution,
            is_done=exp.is_done,
            legal_actions=legal_actions,
            target_value=torch.zeros(self.simulator.num_of_actions + 1),  # dummy value + pi
            info=exp.info
        )

        self.buffer.store_reward(exp.reward, exp.is_done)
        self.current_state = exp.next_state
        return experience

    def update(self):
        if not self.buffer.ready():
            return
        self.buffer.compute_advantages()

        for _ in range(self.ppo_epochs):
            for states, actions, old_log_probs, returns, advantages in self.buffer.iterate_batches():
                logits = self.policy_net(states)
                values = self.value_net(states).squeeze(-1)
                dist = Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(actions)

                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                value_loss = F.mse_loss(values, returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.buffer.reset()
