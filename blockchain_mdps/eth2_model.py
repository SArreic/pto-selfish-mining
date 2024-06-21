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
                                           ['Illegal', 'Propose', 'Attest', 'Vote', 'Wait'])

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

        if action is self.Action.Propose:
            if h > 0:
                next_state = 0, 0, self.Fork.Irrelevant
                transitions.add(next_state, probability=1, difficulty_contribution=h)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Attest:
            if a > h:
                next_state = a - h - 1, 0, self.Fork.Irrelevant
                transitions.add(next_state, probability=1, reward=h + 1, difficulty_contribution=h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Vote:
            if 0 < h <= a < self.max_fork and fork is self.Fork.Relevant:
                next_state = a, h, self.Fork.Active
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        if action is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = a + 1, h, self.Fork.Irrelevant
                transitions.add(attacker_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant
                transitions.add(honest_block, probability=1 - self.alpha)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                attacker_block = a + 1, h, self.Fork.Active
                transitions.add(attacker_block, probability=self.alpha)

                honest_support_block = a - h, 1, self.Fork.Relevant
                transitions.add(honest_support_block, probability=self.gamma * (1 - self.alpha), reward=h,
                                difficulty_contribution=h)

                honest_adversary_block = a, h + 1, self.Fork.Relevant
                transitions.add(honest_adversary_block, probability=(1 - self.gamma) * (1 - self.alpha))
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

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

    def build_malicious_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)

            # Initialize two parallel chains (left chain and right chain)
            left_chain_active = False
            right_chain_active = False

            # Phase 1: Start two parallel chains and keep them private
            if a == 0 and h == 0:
                if fork == self.Fork.Irrelevant:
                    action = self.Action.Propose
                    left_chain_active = True
                elif fork == self.Fork.Relevant:
                    action = self.Action.Propose
                    right_chain_active = True
                else:
                    action = self.Action.Wait

            # Phase 2: Repeat voting on left and right chains
            elif a > 0 and h > 0:
                if left_chain_active and right_chain_active:
                    action = self.Action.Vote

                    action = self.Action.Vote
                elif right_chain_active:
                    action = self.Action.Vote
                else:
                    action = self.Action.Wait

            # Phase 3: Propose new blocks on left or right chain
            elif a > 0:
                if left_chain_active:
                    if fork == self.Fork.Relevant:
                        action = self.Action.Propose
                    else:
                        action = self.Action.Wait
                elif right_chain_active:
                    if fork == self.Fork.Relevant:
                        action = self.Action.Propose
                        action = self.Action.Wait
                else:
                    action = self.Action.Wait

            # Synchronize or delay release of votes
            if a > 0 and h > 0:
                if self.gamma > 0.5:
                    action = self.Action.Attest
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
