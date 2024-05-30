import sys
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Ethereum2Model(BlockchainModel):
    def __init__(self, alpha: float, max_fork: int):
        self.alpha = alpha
        self.max_fork = max_fork

        self.gamma = 0.5
        self.validator_rewards = [1.0, 0.75, 0.5, 0.25]  # Example rewards for validators
        self.inactivity_penalty = 0.5
        self.slashing_penalty = 1.0
        self._reward_dist_b = len(self.validator_rewards)
        self._honest_validators_b = 2 ** (self._reward_dist_b - 1)

        self.Fork = self.create_int_enum('Fork', ['Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Propose', 'Attest', 'Slash', 'Wait'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork)

    def get_state_space(self) -> Space:
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork, 2,
                                                         self._honest_validators_b, self._reward_dist_b)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Relevant, 0, 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Relevant, 0, 0, 0

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, au, hu, r = state

        if action is self.Action.Propose:
            if h > 0:
                new_hu = (hu << 1) % self._honest_validators_b

                proposer_block = 1, 0, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(proposer_block, probability=self.alpha,
                                reward=self.validator_rewards[h % self._reward_dist_b])

                honest_block = 0, 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_block, probability=1 - self.alpha,
                                reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Attest:
            if h > 0:
                new_hu = (hu << 1) % self._honest_validators_b

                attester_block = a + 1, h, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(attester_block, probability=self.alpha,
                                reward=self.validator_rewards[h % self._reward_dist_b])

                honest_block = a, h + 1, self.Fork.Relevant, 0, new_hu, 0
                transitions.add(honest_block, probability=1 - self.alpha,
                                reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Slash:
            if h > 0 and r == 0:
                slashed_block = a, h, self.Fork.Relevant, 0, hu, 0
                transitions.add(slashed_block, probability=self.alpha, reward=-self.slashing_penalty)

                honest_block = a, h, self.Fork.Relevant, 0, hu, 0
                transitions.add(honest_block, probability=1 - self.alpha,
                                reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                proposer_block = a + 1, h, self.Fork.Relevant, 0, hu, r
                transitions.add(proposer_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant, 0, hu, r
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        return transitions

    def build_test_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(1, self.state_space.size):
            a, h, fork, au, hu, r = self.state_space.index_to_element(i)

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

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(1, self.state_space.size):
            a, h, fork, au, hu, r = self.state_space.index_to_element(i)

            if h > 0:
                action = self.Action.Attest
            elif a > 0:
                action = self.Action.Propose
            else:
                action = self.Action.Wait
            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('ethereum2_mdp module test')
    from blockchain_mdps.base.blockchain_mdps.sparse_blockchain_mdp import SparseBlockchainMDP

    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Ethereum2Model(0.35, max_fork=5)
    print(mdp.state_space.size)
    p = mdp.build_honest_policy()

    solver = SparseBlockchainMDP(mdp)
    mdp.print_policy(p, solver.find_reachable_states(p))
    print(solver.calc_policy_revenue(p))
