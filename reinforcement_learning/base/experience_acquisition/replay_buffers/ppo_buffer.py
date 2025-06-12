import torch

from reinforcement_learning.base.experience_acquisition.experience_batch import ExperienceBatch


class PPOBuffer:
    def __init__(self, buffer_size, gamma=0.99, lam=0.95):
        self.legal_actions = []
        self.prev_difficulty_contributions = []
        self.difficulty_contributions = []
        self.legal_actions_list = []
        self.next_states = []
        self.experience_batch = []
        self.advantages = []
        self.returns = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.actions = []
        self.values = []
        self.states = []
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []

        self.next_states = []
        self.legal_actions = []
        self.difficulty_contributions = []
        self.prev_difficulty_contributions = []

    def store(self, state, action, log_prob, value):
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())

    def store_reward(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def ready(self):
        return len(self.rewards) == self.buffer_size

    def compute_advantages(self, device=None):
        self.advantages, self.returns = [], []

        values = self.values + [torch.tensor(0.0)]
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

        self.experience_batch = self.to_experience_batch(device or torch.device('cpu'))

    def iterate_batches(self, batch_size=64):
        self.compute_advantages()
        dataset = list(zip(self.states, self.actions, self.log_probs,
                           self.returns, self.advantages))
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            yield map(torch.stack, zip(*batch))

    def to_experience_batch(self, device: torch.device) -> ExperienceBatch:
        # 使用 value 网络给出值 target_values（TD1 方法）
        with torch.no_grad():
            values = self.values + [torch.tensor(0.0, device=device)]
            rewards = self.rewards
            dones = self.dones

            target_values = []
            for t in range(len(rewards)):
                v_next = values[t + 1]
                v_next = v_next.to(device) if isinstance(v_next, torch.Tensor) else torch.tensor(v_next, device=device)
                td_target = rewards[t] + self.gamma * v_next * (1 - dones[t])
                target_values.append(td_target)

            target_values = torch.stack(target_values).to(device)

        return ExperienceBatch(
            prev_states=torch.stack(self.states).to(device),
            actions=torch.stack(self.actions).to(device),
            next_states=torch.stack(self.next_states).to(device),
            rewards=torch.tensor(self.rewards, device=device),
            difficulty_contributions=torch.stack(self.difficulty_contributions).to(device),
            prev_difficulty_contributions=torch.stack(self.prev_difficulty_contributions).to(device),
            is_done_list=torch.tensor(self.dones, dtype=torch.bool, device=device),
            legal_actions_list=torch.stack(self.legal_actions_list).to(device),
            target_values=target_values
        )

    def store_transition_extra(self, next_state, legal_actions, difficulty, prev_difficulty):
        if not hasattr(self, "next_states"):
            self.next_states = []
            self.legal_actions_list = []
            self.difficulty_contributions = []
            self.prev_difficulty_contributions = []

        self.next_states.append(next_state.detach())
        self.legal_actions_list.append(legal_actions.detach())
        self.difficulty_contributions.append(torch.tensor(difficulty))
        self.prev_difficulty_contributions.append(torch.tensor(prev_difficulty))

