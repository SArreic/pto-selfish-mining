from abc import ABC

from .. import MDPBlockchainSimulator
from ..base.experience_acquisition.agents.bva_agent import BVAAgent
from ..base.function_approximation.approximator import Approximator


class PPOAgent(BVAAgent):
    def __init__(self, actor_critic, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef,
                 approximator: Approximator, simulator: MDPBlockchainSimulator, lr=None, eps=None, max_grad_norm=None):
        super().__init__(approximator, simulator)
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr
        self.eps = eps
        self.max_grad_norm = max_grad_norm

    def update(self, rollouts):
        # Implement PPO update logic here
        pass
