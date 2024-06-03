import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Eth2FeeModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int, fee: float, transaction_chance: float, max_pool: int):
        self.alpha = alpha
        self.gamma = gamma
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance
        self.max_pool = max(max_pool, max_fork)

        self.block_reward = 1

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Propose', 'Attest', 'Slash', 'Wait'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.fee, self.transaction_chance, self.max_pool)

    def get_state_space(self) -> Space:
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool),
                                                                           (0, self.max_fork), (0, self.max_fork),
                                                                           (0, self.max_pool), (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_fork))

    def get_initial_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (0,) * 5

    def get_final_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,) * 5

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int, int, int, int, int]:
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]
        fork = state[-6]
        pool = state[-5]
        length_a = state[-4]
        length_h = state[-3]
        transactions_a = state[-2]
        transactions_h = state[-1]

        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h

    def create_empty_chain(self) -> tuple:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:
        if len(chain) != self.max_fork * 2:
            return False

        valid_parts = sum(isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
                          for block, transaction in zip(chain[::2], chain[1::2]))
        if valid_parts < self.max_fork:
            return False

        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min([self.max_fork - 1] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        invalid_transactions = sum(block is self.Block.NoBlock and transaction is self.Transaction.With
                                   for block, transaction in zip(chain[::2], chain[1::2]))
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        return self.is_chain_valid(a) and self.is_chain_valid(h) \
            and length_a == self.chain_length(a) \
            and length_h == self.chain_length(h) \
            and transactions_a == self.chain_transactions(a) <= pool \
            and transactions_h == self.chain_transactions(h) <= pool

    @staticmethod
    def truncate_chain(chain: tuple, truncate_to: int) -> tuple:
        return chain[:2 * truncate_to]

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def chain_length(self, chain: tuple) -> int:
        return len([block for block in chain[::2] if block is self.Block.Exists])

    def chain_transactions(self, chain: tuple) -> int:
        return len([block for block, transaction in zip(chain[::2], chain[1::2])
                    if block is self.Block.Exists and transaction is self.Transaction.With])

    def add_block(self, chain: tuple, add_transaction: bool) -> tuple:
        transaction = self.Transaction.With if add_transaction else self.Transaction.NoTransaction
        index = self.chain_length(chain)
        chain = list(chain)
        if index < self.max_fork:
            chain[2 * index] = self.Block.Exists
            chain[2 * index + 1] = transaction
        return tuple(chain)

    def propose_block(self, state: BlockchainModel.State, action_param: int) -> tuple:
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        add_transaction = action_param == self.Transaction.With and transactions_a < pool
        new_chain = self.add_block(a, add_transaction)
        return new_chain

    def apply_block(self, state: BlockchainModel.State, new_block: tuple) -> BlockchainModel.State:
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        new_length_a = self.chain_length(new_block)
        new_transactions_a = self.chain_transactions(new_block)
        return new_block + h + (fork,) + (pool,) + (new_length_a,) + (length_h,) + (new_transactions_a,) + (transactions_h,)

    def is_valid_attestation(self, state: BlockchainModel.State, action_param: int) -> bool:
        return True

    def apply_attestation(self, state: BlockchainModel.State, action_param: int) -> BlockchainModel.State:
        return state

    def is_slashable(self, state: BlockchainModel.State, action_param: int) -> bool:
        return True

    def apply_slash(self, state: BlockchainModel.State, action_param: int) -> BlockchainModel.State:
        return state

    def is_valid_aggregation(self, state: BlockchainModel.State, action_param: int) -> bool:
        return True

    def apply_aggregation(self, state: BlockchainModel.State, action_param: int) -> BlockchainModel.State:
        return state

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        action_type, action_param = action

        if action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        else:
            if action_type is self.Action.Propose:
                new_block = self.propose_block(state, action_param)
                if self.is_chain_valid(new_block):
                    new_state = self.apply_block(state, new_block)
                    reward = self.block_reward + (self.fee if action_param == self.Transaction.With else 0)
                    transitions.add(new_state, probability=1, reward=reward)
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif action_type is self.Action.Attest:
                if self.is_valid_attestation(state, action_param):
                    new_state = self.apply_attestation(state, action_param)
                    reward = self.block_reward / 2
                    transitions.add(new_state, probability=1, reward=reward)
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif action_type is self.Action.Slash:
                if self.is_slashable(state, action_param):
                    new_state = self.apply_slash(state, action_param)
                    reward = self.block_reward / 2
                    transitions.add(new_state, probability=1, reward=reward)
                else:
                    transitions.add(self.final_state, probability=1, reward=self.error_penalty)

            elif action_type is self.Action.Wait:
                transitions.add(state, probability=1, reward=0)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha * self.block_reward * (1 + self.fee * self.transaction_chance)


if __name__ == '__main__':
    print('eth2_fee_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Eth2FeeModel(0.35, 0.5, 2, fee=2, transaction_chance=0.1, max_pool=2)
    print(mdp.state_space.size)
