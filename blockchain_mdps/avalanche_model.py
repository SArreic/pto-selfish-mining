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
    def __init__(self, alpha: float, gamma: float, clock: int, max_forks: int, fee: float, pool: float):
        self.alpha = alpha  # Honest validators proportion
        self.gamma = gamma  # Attacker proportion
        self.max_forks = max_forks  # Maximum number of fork chains per user
        self.max_sys_forks = int(max_forks * 1e1)
        self.fee = fee  # tax of block reward
        self.pool = pool
        self.clock = clock

        self.Action = self.create_int_enum('Action', ['Illegal', 'AddBlock', 'CreateFork'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_forks})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_forks)

    def get_state_space(self) -> Space:
        # State: (main_chain_depth, active_chain_index, num_forks)
        underlying_space = MultiDimensionalDiscreteSpace(
            (0, int(self.max_forks)),  # Number of user forks
            (0, self.max_sys_forks),  # Number of system forks
            (0, int(self.clock))  # Maximum number of security reward pool
        )
        print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        action_space = DiscreteSpace(self.Action)
        print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        # Start with main chain depth of 0, active chain is main, no forks
        return 0, 0, self.clock

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, -1

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        num_forks, sys_num_forks, clock = state

        weight = num_forks / (sys_num_forks + 1)

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if clock == 0:
            transitions.add((0, 0, self.clock), probability=1, reward=0)
            return transitions

        if action is self.Action.AddBlock:
            self.pool += self.fee
            if num_forks >= self.max_forks or num_forks < 0:
                # Chain has reached maximum depth, user must publish
                transitions.add(self.final_state, probability=1, reward=0)
            else:
                transitions.add((num_forks, sys_num_forks, clock - 1),
                                probability=weight,
                                reward=self.clock - clock - self.fee)

                num_forks = min(num_forks + 1, self.max_forks)
                # Only Basic reward for adding block
                transitions.add((num_forks, min(sys_num_forks + 1, self.max_sys_forks), clock - 1),
                                probability=1 - weight,
                                reward=1 - self.fee  # Set reward to 1 - tax
                                )

        elif action is self.Action.CreateFork:
            reward = max(self.pool * (sys_num_forks - num_forks) / (sys_num_forks + 1), 0)
            self.pool -= reward
            sys_num_forks = min(sys_num_forks + 1, self.max_sys_forks)
            if 0 <= num_forks < self.max_forks:
                # No reward for creating fork
                transitions.add(
                    (num_forks + 1, sys_num_forks, clock - 1),  # Creating a new fork
                    probability=1,
                    reward=reward + 1 - self.fee
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
