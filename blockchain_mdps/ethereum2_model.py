import sys
from enum import Enum
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
        # alpha: max shards, max_fork: max validators per shard

        self.gamma = 0.5
        self.validator_rewards = [1.0, 0.75, 0.5, 0.25]  # Example rewards for validators
        self.inactivity_penalty = 0.5
        self.slashing_penalty = 1.0
        self._reward_dist_b = len(self.validator_rewards)
        self._honest_validators_b = 2 ** (self._reward_dist_b - 1)

        self.Fork = self.create_int_enum('Fork', ['Relevant', 'Active'])
        self.Action = self.create_int_enum('Action',
                                           ['Illegal', 'Propose', 'Attest', 'Slash', 'Wait'])
        self.Shard = self.create_int_enum('Shard', ['Inactive', 'Active'])
        self.Validator = self.create_int_enum('Validator', ['InPool', 'Active'])
        self.Epoch = self.create_int_enum('Epoch', ['Inactive', 'Active'])
        self.Slot = self.create_int_enum('Slot', ['Empty', 'Filled'])

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

    def get_initial_state(self) -> Tuple:
        initial_shards = (self.Shard.Inactive,) * self.alpha
        initial_validators = ((self.Validator.InPool,) * self.max_fork,) * self.alpha
        return initial_shards + initial_validators

    def get_final_state(self) -> Tuple:
        final_shards = (self.Shard.Active,) * self.alpha
        final_validators = ((self.Validator.Active,) * self.max_fork,) * self.alpha
        return final_shards + final_validators

    # def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int, int, int]:
    #     a = state[:2 * self.max_validators_per_shard]
    #     h = state[2 * self.max_validators_per_shard:4 * self.max_validators_per_shard]
    #     fork = state[-6]
    #     au = state[-5]
    #     hu = state[-4]
    #     r = state[-3]
    #     return a, h, fork, au, hu, r

    def dissect_state(self, state: Tuple) -> Tuple:
        shards = state[:self.alpha]
        validators_per_shard = state[self.alpha:]
        return shards, validators_per_shard

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, au, hu, r = self.dissect_state(state)

        if action is self.Action.Propose:
            if h > 0:
                new_hu = (hu << 1) % self._honest_validators_b

                proposer_block = 1, 0, self.Fork.Relevant, 0, new_hu, 0
                if proposer_block in transitions.probabilities:
                    transitions.probabilities[proposer_block] += self.alpha
                    transitions.rewards[proposer_block] += self.validator_rewards[h % self._reward_dist_b]
                else:
                    transitions.add(proposer_block, probability=self.alpha,
                                    reward=self.validator_rewards[h % self._reward_dist_b])

                honest_block = 0, 1, self.Fork.Relevant, 0, new_hu, 0
                if honest_block in transitions.probabilities:
                    transitions.probabilities[honest_block] += 1 - self.alpha
                    transitions.rewards[honest_block] += self.validator_rewards[h % self._reward_dist_b]
                else:
                    transitions.add(honest_block, probability=1 - self.alpha,
                                    reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Attest:
            if h > 0:
                new_hu = (hu << 1) % self._honest_validators_b

                attester_block = a + 1, h, self.Fork.Relevant, 0, new_hu, 0
                if attester_block in transitions.probabilities:
                    transitions.probabilities[attester_block] += self.alpha
                    transitions.rewards[attester_block] += self.validator_rewards[h % self._reward_dist_b]
                else:
                    transitions.add(attester_block, probability=self.alpha,
                                    reward=self.validator_rewards[h % self._reward_dist_b])

                honest_block = a, h + 1, self.Fork.Relevant, 0, new_hu, 0
                if honest_block in transitions.probabilities:
                    transitions.probabilities[honest_block] += 1 - self.alpha
                    transitions.rewards[honest_block] += self.validator_rewards[h % self._reward_dist_b]
                else:
                    transitions.add(honest_block, probability=1 - self.alpha,
                                    reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Slash:
            if h > 0 and r == 0:
                slashed_block = a, h, self.Fork.Relevant, 0, hu, 0
                if slashed_block in transitions.probabilities:
                    transitions.probabilities[slashed_block] += self.alpha
                    transitions.rewards[slashed_block] -= self.slashing_penalty
                else:
                    transitions.add(slashed_block, probability=self.alpha, reward=-self.slashing_penalty)

                honest_block = a, h, self.Fork.Relevant, 0, hu, 0
                if honest_block in transitions.probabilities:
                    transitions.probabilities[honest_block] += 1 - self.alpha
                    transitions.rewards[honest_block] += self.validator_rewards[h % self._reward_dist_b]
                else:
                    transitions.add(honest_block, probability=1 - self.alpha,
                                    reward=self.validator_rewards[h % self._reward_dist_b])
            else:
                transitions.add(self.final_state, probability=1, reward=self.inactivity_penalty)

        if action is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                proposer_block = a + 1, h, self.Fork.Relevant, 0, hu, r
                if proposer_block in transitions.probabilities:
                    transitions.probabilities[proposer_block] += self.alpha
                else:
                    transitions.add(proposer_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant, 0, hu, r
                if honest_block in transitions.probabilities:
                    transitions.probabilities[honest_block] += 1 - self.alpha
                else:
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

    # def get_honest_revenue(self) -> float:
    #     return self.alpha

    def get_honest_revenue(self) -> float:
        """
        Calculate the expected revenue of an honest validator using the honest policy.
        """
        honest_policy = self.build_honest_policy()
        solver = SparseBlockchainMDP(self)
        revenue = solver.calc_policy_revenue(honest_policy)
        return revenue


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

    # Calculate and print honest revenue
    honest_revenue = mdp.get_honest_revenue()
    print(f'Honest Revenue: {honest_revenue}')

