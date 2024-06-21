import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Ethereum2Model(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_proposals: int, max_votes: int, max_stake_pool: int):
        self.alpha = alpha
        self.gamma = gamma
        self.max_proposals = max_proposals
        self.max_votes = max_votes
        self.max_stake_pool = max_stake_pool

        self.Validator = self.create_int_enum('Validator', ['Active', 'Pending', 'Slashed'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Attest', 'Propose', 'Vote', 'Wait'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_proposals}, {self.max_votes}, {self.max_stake_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_proposals, self.max_votes, self.max_stake_pool)

    def get_state_space(self) -> Space:
        elements = [(0, self.max_proposals), (0, self.max_votes), self.Validator, (0, self.max_stake_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_proposals))

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Validator.Active, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Validator.Slashed, -1

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[int, int, Enum, int]:
        proposals = state[0]
        votes = state[1]
        validator_state = state[2]
        stake_pool = state[3]
        # print( f'Dissecting state: proposals={proposals}, votes={votes}, validator_state={validator_state},
        # stake_pool={stake_pool}')
        return proposals, votes, validator_state, stake_pool

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        proposals, votes, validator_state, stake_pool = self.dissect_state(state)
        action_type, action_param = action

        if action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action_type is self.Action.Attest:
            if validator_state == self.Validator.Active and votes < self.max_votes:
                next_state = (proposals, votes + 1, validator_state, stake_pool)
                transitions.add(next_state, probability=1)

        if action_type is self.Action.Propose:
            if validator_state == self.Validator.Active and proposals < self.max_proposals:
                next_state = (proposals + 1, votes, validator_state, stake_pool)
                transitions.add(next_state, probability=1)

        if action_type is self.Action.Vote:
            if validator_state == self.Validator.Active and action_param <= proposals and votes < self.max_votes:
                next_state = (proposals, votes + 1, validator_state, stake_pool)
                transitions.add(next_state, probability=1)

        if action_type is self.Action.Wait:
            next_state = state
            transitions.add(next_state, probability=1)

        return transitions

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        proposals, votes, validator_state, stake_pool = self.dissect_state(state)
        return proposals >= 0 and votes >= 0 and validator_state in self.Validator and stake_pool >= 0

    def get_honest_revenue(self) -> float:
        return self.alpha

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            proposals, votes, validator_state, stake_pool = self.state_space.index_to_element(i)

            if proposals > votes:
                action = self.Action.Propose
            elif votes > proposals:
                action = self.Action.Vote
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)

    def build_attack_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            proposals, votes, validator_state, stake_pool = self.state_space.index_to_element(i)

            if proposals > votes + 1:
                action = self.Action.Propose
            elif (votes == proposals - 1 and proposals >= 2) or proposals == self.max_proposals:
                action = self.Action.Vote
            elif (votes == 1 and proposals == 1) and validator_state is self.Validator.Active:
                action = self.Action.Attest
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('ethereum_model module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Ethereum2Model(0.35, 0.5, 10, 10, 100)
    print(mdp.get_state_space().size)
