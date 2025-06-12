import torch
import torch.nn as nn
import torch.nn.functional as F

def _build_net(in_dim, out_dim, hidden_dims):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class PPOApproximator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.policy_net = _build_net(state_dim, action_dim, hidden_dims)
        self.value_net = _build_net(state_dim, 1, hidden_dims)

    def forward(self, state):
        logits = self.policy_net(state)
        value = self.value_net(state).squeeze(-1)
        return logits, value


class CompatibleApproximatorWrapper:
    """
    适配器：用于统一 PPOApproximator 接口与 WeRLman 框架预期格式
    """
    def __init__(self, ppo_approximator):
        self.approximator = ppo_approximator
        self.target_values = None

    def __call__(self, state):
        logits, value = self.approximator(state)
        probs = torch.softmax(logits, dim=-1)
        self.target_values = torch.cat([value.view(1), probs.view(-1)])
        return self

    @property
    def policy_net(self):
        return self.approximator.policy_net

    @property
    def value_net(self):
        return self.approximator.value_net
