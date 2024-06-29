import math
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

        # Define the rewards for different actions
        self.propose_reward = 10  # Placeholder, set according to Ethereum 2.0 specifics
        self.attest_reward = 5  # Placeholder, set according to Ethereum 2.0 specifics
        self.vote_reward = 3  # Placeholder, set according to Ethereum 2.0 specifics

        self.User = self.create_int_enum('User', ['Proposer', 'Committee', 'Validator'])
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
            self.User, (0, self.max_stake_pool), (0, self.max_pool), (0, self.max_fee_pool),
            (0, self.network_congestion)
        ]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_proposals))

    def get_initial_state(self) -> BlockchainModel.State:
        initial_state = (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_proposals * 2 + (
            self.User.Validator, 0, 0, 0, 0)
        return initial_state

    def get_final_state(self) -> BlockchainModel.State:
        final_state = (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_proposals * 2 + (
            self.User.Validator, -1, -1, -1, -1)
        return final_state

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
        expected_blocks_length = self.max_proposals * 4

        if len(new_blocks) < expected_blocks_length:
            additional_blocks_needed = (expected_blocks_length - len(new_blocks)) // 2
            new_blocks.extend([self.Block.NoBlock, self.Transaction.NoTransaction] * additional_blocks_needed)

        new_blocks[2 * index] = self.Block.Exists
        new_blocks[2 * index + 1] = self.Transaction.NoTransaction

        new_state = tuple(new_blocks) + (validator_state, stake_pool, pool, fee_pool, congestion)

        return new_state

    def add_vote(self, state: BlockchainModel.State, proposal_index: int) -> BlockchainModel.State:
        blocks, transactions, validator_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        if proposal_index < 0 or proposal_index > self.max_proposals:
            raise ValueError(f"Invalid proposal_index: {proposal_index}")
        index = sum(1 for block in blocks if block is self.Block.Exists)
        if index <= proposal_index:
            return state
        new_transactions = list(transactions)
        while len(new_transactions) <= 2 * proposal_index:
            new_transactions.append(self.Transaction.NoTransaction)
        new_transactions[2 * proposal_index] = self.Transaction.WithTransaction
        new_state = blocks + tuple(new_transactions) + (validator_state, stake_pool, pool, fee_pool, congestion)
        return new_state

    def get_reward(self, state: BlockchainModel.State, reward_factor: float) -> float:
        blocks, transactions, user_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        base_reward = ((stake_pool * 6.4) / (0.4 * math.sqrt(1e3)))
        max_possible_reward = (self.max_stake_pool * 6.4) / (0.4 * math.sqrt(1e3))
        normalized_reward = base_reward * reward_factor / max_possible_reward
        return normalized_reward

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        blocks, transactions, user_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        action_type, action_param = action

        if action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
        elif action_type is self.Action.Propose:
            if user_state == self.User.Proposer:
                next_state = self.add_proposal(state)
                reward = self.get_reward(next_state, self.propose_reward)
                transitions.add(next_state, probability=1, reward=reward)
        elif action_type is self.Action.Vote:
            if user_state in [self.User.Committee, self.User.Validator] and action_param <= self.max_proposals:
                next_state = self.add_vote(state, action_param)
                reward = self.get_reward(next_state, self.vote_reward)
                transitions.add(next_state, probability=1, reward=reward)
        elif action_type is self.Action.Attest:
            if user_state == self.User.Validator:
                next_state = state  # Placeholder for Attest action logic if needed
                reward = self.get_reward(next_state, self.attest_reward)
                transitions.add(next_state, probability=1, reward=reward)

        # Ensure total transition probabilities sum to 1
        total_prob = sum(transitions.probabilities.values())
        if total_prob == 0:
            num_transitions = len(transitions.probabilities)
            if num_transitions == 0:
                transitions.add(self.final_state, probability=1)
                total_prob = 1
            else:
                for key in transitions.probabilities.keys():
                    transitions.probabilities[key] = 1 / num_transitions
                total_prob = sum(transitions.probabilities.values())
                if not math.isclose(total_prob, 1, abs_tol=1e-5):
                    raise ValueError("Total transition probability does not sum to 1 after adjustment")

        if not math.isclose(total_prob, 1, abs_tol=1e-5):
            factor = 1 / total_prob
            for key in transitions.probabilities.keys():
                transitions.probabilities[key] *= factor
            total_prob = sum(transitions.probabilities.values())
            if not math.isclose(total_prob, 1, abs_tol=1e-5):
                remaining_prob = 1 - total_prob
                for key in transitions.probabilities.keys():
                    if remaining_prob != 0:
                        transitions.probabilities[key] += remaining_prob / len(transitions.probabilities)
                    total_prob = sum(transitions.probabilities.values())
                    if not math.isclose(total_prob, 1, abs_tol=1e-5):
                        raise ValueError("Total transition probability does not sum to 1 after adjustment")

        return transitions

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        blocks, transactions, user_state, stake_pool, pool, fee_pool, congestion = self.dissect_state(state)
        if stake_pool < 0 or pool < 0 or fee_pool < 0 or congestion < 0:
            return False
        if len(blocks) != 2 * self.max_proposals:
            return False
        if len(transactions) != 2 * self.max_proposals:
            return False
        if user_state not in self.User:
            return False
        return True

    def get_honest_revenue(self) -> float:
        return self.alpha


if __name__ == '__main__':
    print('ethereum_fee_model module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Ethereum2FeeModel(0.35, 0.5, 10, 10, 100, 50, 20, 5)
    print(mdp.get_state_space().size)
