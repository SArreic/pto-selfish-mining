import torch
import torch.nn as nn


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
