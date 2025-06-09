import torch
import torch.nn.functional as F

class PPOLossFunction:
    def __init__(self, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute(self, old_log_probs, new_log_probs, values, returns, advantages, entropy):
        ratio = (new_log_probs - old_log_probs).exp()
        clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return loss
