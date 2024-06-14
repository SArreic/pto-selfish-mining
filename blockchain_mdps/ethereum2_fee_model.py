import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Ethereum2FeeModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_proposals: int, max_votes: int, max_stake_pool: int,
                 max_pool: int, max_fee_pool: int, network_congestion: int):
        self.alpha = alpha
        self.gamma = gamma
        self.max_proposals = max_proposals
        self.max_votes = max_votes
        self.max_stake_pool = max_stake_pool
        self.max_pool = max_pool
        self.max_fee_pool = max_fee_pool
        self.network_congestion = network_congestion

        self.Validator = self.create_int_enum('Validator', ['Active', 'Pending', 'Slashed'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Propose', 'Vote', 'Attest'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'WithTransaction'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_proposals}, {self.max_votes}, {self.max_stake_pool}, {self.max_pool}, {self.max_fee_pool}, {self.network_congestion})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (
            self.alpha, self.gamma, self.max_proposals, self.max_votes, self.max_stake_pool, self.max_pool,
            self.max_fee_pool, self.network_congestion)

    def get_state_space(self) -> Space:
        elements = [self.Block, self.Transaction] * (2 * self.max_proposals) + [
            self.Validator, (0, self.max_stake_pool), (0, self.max_pool), (0, self.max_fee_pool),
            (0, self.network_congestion)
        ]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_proposals))

    def get_initial_state(self) -> BlockchainModel.State:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_proposals + (
            self.Validator.Active, 0, 0, 0, 0)

    def get_final_state(self) -> BlockchainModel.State:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_proposals + (
            self.Validator.Slashed, -1, -1, -1, -1)

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int, int, int, int]:
        if len(state) != 4 * self.max_proposals + 5:
            raise ValueError(f"State length {len(state)} does not match expected {4 * self.max_proposals + 5}")

        blocks = state[:2 * self.max_proposals]
        transactions = state[2 * self.max_proposals:4 * self.max_proposals]
        validator_state = state[4 * self.max_proposals]
        stake_pool = state[4 * self.max_proposals + 1]
        pool = state[4 * self.max_proposals + 2]
        fee_pool = state[4 * self.max_proposals + 3]
        congestion = state[4 * self.max_proposals + 4]

        return blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion

    def add_proposal(self, state: BlockchainModel.State) -> BlockchainModel.State:
        blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        index = sum(1 for block in blocks if block is self.Block.Exists)
        new_blocks = list(blocks)
        new_blocks[2 * index] = self.Block.Exists
        new_blocks[2 * index + 1] = self.Transaction.NoTransaction
        return tuple(new_blocks) + (validator_state, stake_pool, pool, fee_pool, congestion)

    def add_vote(self, state: BlockchainModel.State, proposal_index: int) -> BlockchainModel.State:
        blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        index = sum(1 for block in blocks if block is self.Block.Exists)
        if index >= proposal_index:
            return state  # Invalid vote if trying to vote on non-existing proposal
        new_transactions = list(transactions)
        new_transactions[2 * proposal_index] = self.Transaction.WithTransaction
        return blocks + tuple(new_transactions) + (validator_state, stake_pool, pool, fee_pool, congestion)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        action_type, action_param = action

        if action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action_type is self.Action.Propose:
            if validator_state == self.Validator.Active:
                next_state = self.add_proposal(state)
                transitions.add(next_state, probability=1)

        if action_type is self.Action.Vote:
            if validator_state == self.Validator.Active and action_param <= self.max_proposals:
                next_state = self.add_vote(state, action_param)
                transitions.add(next_state, probability=1)

        if action_type is self.Action.Attest:
            if validator_state == self.Validator.Active:
                next_state = state  # Placeholder for Attest action logic if needed
                transitions.add(next_state, probability=1)

        return transitions

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        return len(blocks) == 2 * self.max_proposals and len(transactions) == 2 * self.max_proposals \
            and all(isinstance(block, self.Block) for block in blocks) \
            and all(isinstance(transaction, self.Transaction) for transaction in transactions) \
            and validator_state in self.Validator \
            and 0 <= stake_pool <= self.max_stake_pool \
            and 0 <= pool <= self.max_pool \
            and 0 <= fee_pool <= self.max_fee_pool \
            and 0 <= congestion <= self.network_congestion

    def get_honest_revenue(self) -> float:
        return self.alpha


if __name__ == '__main__':
    print('ethereum_fee_model module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Ethereum2FeeModel(0.35, 0.5, 10, 10, 100, 50, 20, 5)
    print(mdp.get_state_space().size)
