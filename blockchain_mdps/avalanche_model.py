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
    def __init__(self, alpha: float, beta: float, max_depth: int, max_forks: int, tax: float, pool: float):
        self.max_depth = max_depth  # Maximum depth for
        self.alpha = alpha  # Honest validators proportion
        self.beta = beta  # Attacker proportion
        self.max_forks = max_forks  # Maximum number of fork chains per user
        self.tax = tax  # tax of block reward
        self.max_pool = pool

        self.Action = self.create_int_enum('Action', ['AddBlock', 'CreateFork'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.beta}, {self.max_forks})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.beta, self.max_forks)

    def get_state_space(self) -> Space:
        # State: (main_chain_depth, active_chain_index, num_forks)
        underlying_space = MultiDimensionalDiscreteSpace(
            (0, self.max_depth),  # User main chain depth
            (0, self.max_forks),  # Number of user forks
            (0, self.max_depth * 1e3),  # System main chain depth
            (0, self.max_forks * 1e2),  # Number of system forks
            (0, self.max_pool)  # Maximum number of security reward pool
        )
        print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        action_space = DiscreteSpace(self.Action)
        print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        # Start with main chain depth of 0, active chain is main, no forks
        return 0, 0, 0, 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, -1, -1, -1

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        main_chain_depth, num_forks, sys_main_chain_depth, sys_num_forks, reward_pool = state

        weight = (num_forks * 0.8 + main_chain_depth * 0.5) / (main_chain_depth + num_forks)

        if action is self.Action.AddBlock:
            new_num_forks = max(num_forks - 1, 0)
            new_pool = min(reward_pool + self.tax, self.max_pool)
            if main_chain_depth >= self.max_depth or main_chain_depth <= 0:
                # Chain has reached maximum depth, user must publish
                transitions.add(self.final_state, probability=1, reward=0)
            else:
                transitions.add((main_chain_depth, new_num_forks, sys_main_chain_depth + 1, sys_num_forks,
                                 new_pool),
                                probability=weight, reward=main_chain_depth + 1 - self.tax)

                # Only Basic reward for adding block
                transitions.add((main_chain_depth + 1, new_num_forks, sys_main_chain_depth, sys_num_forks,
                                 new_pool),
                                probability=1 - weight,
                                reward=1 - self.tax  # Set reward to 1 - tax
                                )

        elif action is self.Action.CreateFork:
            if num_forks > 0 and main_chain_depth < 1:
                main_chain_depth += 1
                num_forks -= 1
            new_main_chain_depth = max(main_chain_depth - 1, 0)
            reward = reward_pool * (sys_main_chain_depth / (sys_main_chain_depth + sys_num_forks))
            if num_forks < self.max_forks and main_chain_depth > 1:
                # No reward for creating fork
                transitions.add(
                    (new_main_chain_depth, num_forks + 1, sys_main_chain_depth, sys_num_forks + 1,
                     reward_pool - reward),  # Creating a new fork
                    probability=1,
                    reward=reward
                )
            else:
                transitions.add(self.final_state, probability=1, reward=0)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        initial_state = (0, 0, 0)
        return policy[self.state_space.element_to_index(initial_state)] == self.Action.AddBlock

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            action = self.Action.AddBlock

            policy[i] = action

        return tuple(policy)

    def build_attack_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            if active_chain != 0:
                action = self.Action.CreateFork
            else:
                action = self.Action.AddBlock

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('simplified_avalanche_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = AvalancheAttackModel(0.6, 0.4, 3)
    print(mdp.state_space.size)
    p = mdp.build_attack_policy()
    print(p[:10])
