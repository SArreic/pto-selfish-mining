import torch

class PPOBuffer:
    def __init__(self, buffer_size, gamma=0.99, lam=0.95):
        self.advantages = None
        self.returns = None
        self.rewards = None
        self.dones = None
        self.log_probs = None
        self.actions = None
        self.values = None
        self.states = None
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        self.states, self.actions, self.log_probs = [], [], []
        self.values, self.rewards, self.dones = [], [], []

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

    def compute_advantages(self):
        self.advantages, self.returns = [], []
        values = self.values + [torch.tensor(0.0)]
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - self.dones[t]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

    def iterate_batches(self, batch_size=64):
        self.compute_advantages()
        dataset = list(zip(self.states, self.actions, self.log_probs,
                           self.returns, self.advantages))
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            yield map(torch.stack, zip(*batch))
