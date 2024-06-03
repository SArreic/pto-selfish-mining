from abc import ABC
from typing import Optional

import torch

from .caching_agent import CachingAgent
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator


class ACAgent(CachingAgent, ABC):

    def __init__(self, actor_approximator: Approximator, critic_approximator: Approximator,
                 simulator: MDPBlockchainSimulator, use_cache: bool = True):
        super().__init__(actor_approximator, critic_approximator, simulator, use_cache)
        self.base_value_approximation = 0

    def actor_update(self, actor_approximator: Optional[Approximator] = None,
                     base_value_approximation: Optional[float] = None,
                     **kwargs) -> None:
        super().update(actor_approximator, **kwargs)

        if base_value_approximation is not None:
            self.base_value_approximation = base_value_approximation

    def critic_update(self, critic_approximator: Optional[Approximator] = None,
                      base_value_approximation: Optional[float] = None,
                      **kwargs) -> None:
        super().update(critic_approximator, **kwargs)

        if base_value_approximation is not None:
            self.base_value_approximation = base_value_approximation

    def reduce_to_v_table(self) -> torch.Tensor:
        v_table = super().reduce_to_v_table()
        if hasattr(self, 'nn_factor'):
            v_table *= self.nn_factor

        v_table += self.base_value_approximation

        return v_table
