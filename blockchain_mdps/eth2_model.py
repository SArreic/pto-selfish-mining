import sys
from typing import Tuple

import numpy as np

from blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Eth2Model(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int):
        self.alpha = alpha
        self.gamma = gamma
        self.max_fork = max_fork

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action',
                                           ['Illegal', 'Propose', 'Attest', 'Slash', 'Wait'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork)

    def get_state_space(self) -> Space:
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Irrelevant

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Irrelevant

    # noinspection DuplicatedCode
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork = state

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        elif action is self.Action.Propose:
            if a + 1 <= self.max_fork:
                next_state = a + 1, h, self.Fork.Relevant
                transitions.add(next_state, probability=1)
            else:
                transitions.add(state, probability=1, reward=self.error_penalty)

        elif action is self.Action.Attest:
            if h + 1 <= self.max_fork:
                next_state = a, h + 1, self.Fork.Relevant
                transitions.add(next_state, probability=1)
            else:
                transitions.add(state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Slash:
            if fork == self.Fork.Active:
                new_state = self.get_final_state()
                transitions.add(new_state, probability=1)
            else:
                transitions.add(state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Wait:
            transitions.add(state, probability=1)

        return transitions

    def get_honest_revenue(self) -> float:
        """
        Calculate the expected revenue of an honest validator using the honest policy.
        """
        honest_policy = self.build_honest_policy()
        solver = SparseBlockchainMDP(self)
        revenue = solver.calc_policy_revenue(honest_policy)
        return revenue

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)

            if h > 0:
                action = self.Action.Attest
            elif a > 0:
                action = self.Action.Propose
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)

    def build_test_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(1, self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)

            if h > 1:
                action = self.Action.Propose
            elif h > a:
                action = self.Action.Attest
            elif a > 0:
                action = self.Action.Slash
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('ethereum_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Eth2Model(0.35, 0.5, 100)
    print(mdp.state_space.size)
    p = mdp.build_test_policy()
    print(p[:10])
