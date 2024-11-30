from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch

from blockchain_mdps import BlockchainModel
from ..agents.planning_agent import PlanningAgent
from ...blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
from ...function_approximation.approximator import Approximator


class CachingAgent(PlanningAgent, ABC):
    def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, use_cache: bool = True):
        super().__init__(approximator, simulator)
        self.state_value_cache = {}
        self.use_cache = use_cache

    def flatten_state(self, state) -> list:
        """递归展平嵌套结构."""
        flat_state = []
        if isinstance(state, (tuple, list)):
            for element in state:
                flat_state.extend(self.flatten_state(element))  # 递归展平每个元素
        else:
            flat_state.append(state)  # 直接添加其他类型元素
        return flat_state

    def get_state_evaluation(self, state: BlockchainModel.State, exploring: bool) -> torch.Tensor:
        print("Type of state: ", type(state))
        print("Type of state element is: ", type(state[0]))
        print("State is: ", state)
        print("Length of original state is : ", len(state))
        if state not in self.state_value_cache or not self.use_cache:
            flatten_state = self.flatten_state(state)
            print("Type of flattened state: ", type(flatten_state))
            print("Type of flattened state element is: ", type(flatten_state[0]))
            print("Flattened State is: ", flatten_state)
            print("Length of flattened state is : ", len(flatten_state))
            state_tensor = torch.tensor(
                flatten_state,
                device=self.simulator.device,
                dtype=torch.float,
            )
            state_eval = self.evaluate_state(state_tensor, exploring)

            if self.use_cache:
                self.state_value_cache[state] = state_eval
            else:
                return state_eval

        return self.state_value_cache[state]

    @abstractmethod
    def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
        pass

    def update(self, approximator: Optional[Approximator] = None, **kwargs) -> None:
        super().update(approximator, **kwargs)
        self.state_value_cache = {}
