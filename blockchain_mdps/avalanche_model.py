import sys
from typing import Tuple

import numpy as np

from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class AvalancheAttackModel(BlockchainModel):
    def __init__(self, alpha: float, beta: float, max_depth: int, max_forks: int):
        self.alpha = alpha  # Honest validators proportion
        self.beta = beta  # Attacker proportion
        self.max_depth = max_depth  # Maximum depth of any chain
        self.max_forks = max_forks  # Maximum number of fork chains

        self.Action = self.create_int_enum('Action', ['AddBlock', 'CreateFork', 'SwitchChain', 'PublishFork'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.beta}, {self.max_depth}, {self.max_forks})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.beta, self.max_depth, self.max_forks)

    def get_state_space(self) -> Space:
        # State: (chain_depths, main_chain_index, num_forks, total_weight)
        underlying_space = MultiDimensionalDiscreteSpace(
            *[(0, self.max_depth)] * (self.max_forks + 1),  # Depth of each chain (including main chain)
            (0, self.max_forks),  # Active chain index (0 = main chain, 1 to max_forks = fork chains)
            (0, self.max_forks),  # Number of forks
            (0, self.max_depth * (self.max_forks + 1))  # Total weight (sum of all chain depths)
        )
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        # Start with main chain depth of 0, active chain is main, no forks, total weight = 0
        return tuple(0 for _ in range(self.max_forks + 1)) + (0, 0, 0)

    def get_final_state(self) -> BlockchainModel.State:
        return tuple(-1 for _ in range(self.max_forks + 1)) + (-1, -1, -1)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        *chain_depths, active_chain, num_forks, total_weight = state
        new_chain_depths = list(chain_depths)

        if action is self.Action.AddBlock:
            new_chain_depths[active_chain] += 1
            new_total_weight = total_weight + 1
            transitions.add(
                (*new_chain_depths, active_chain, num_forks, new_total_weight),
                probability=self.alpha if active_chain == 0 else self.beta,
                reward=1
            )

        elif action is self.Action.CreateFork:
            if num_forks < self.max_forks:
                new_chain_depths[num_forks + 1] = 1  # New fork with depth 1
                new_total_weight = total_weight + 1
                transitions.add(
                    (*new_chain_depths, num_forks + 1, num_forks + 1, new_total_weight),
                    probability=self.beta,
                    reward=0.5
                )

        elif action is self.Action.SwitchChain:
            best_chain = 0  # Default to main chain
            best_depth = chain_depths[0]
            for i, depth in enumerate(chain_depths[1:], start=1):
                if depth > best_depth:
                    best_chain, best_depth = i, depth

            if best_chain != active_chain:
                transitions.add(
                    (*chain_depths, best_chain, num_forks, total_weight),
                    probability=self.beta,
                    reward=1
                )

        elif action is self.Action.PublishFork:
            reward = 0.75
            transitions.add(
                (*chain_depths, active_chain, num_forks, total_weight),
                probability=1,
                reward=reward
            )

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        initial_state = tuple(0 for _ in range(self.max_forks + 1)) + (0, 0, 0)
        return policy[self.state_space.element_to_index(initial_state)] == self.Action.AddBlock

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            *chain_depths, active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            if active_chain == 0 or chain_depths[active_chain] <= max(chain_depths):
                action = self.Action.AddBlock
            else:
                action = self.Action.SwitchChain

            policy[i] = action

        return tuple(policy)

    def build_attack_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            *chain_depths, active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            if active_chain != 0 and chain_depths[active_chain] > chain_depths[0]:
                action = self.Action.SwitchChain
            elif num_forks < self.max_forks:
                action = self.Action.CreateFork
            else:
                action = self.Action.PublishFork

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('avalanche_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = AvalancheAttackModel(0.6, 0.4, 10, 3)
    print(mdp.state_space.size)
    p = mdp.build_attack_policy()
    print(p[:10])
