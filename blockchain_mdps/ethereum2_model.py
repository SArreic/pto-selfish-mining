import math
import sys
from enum import Enum
from random import random
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Ethereum2Model(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_proposals: int, max_votes: int,
                 max_stake_pool: float):
        self.alpha = alpha if alpha is not None else 0.3  # Proportion of malicious users
        self.gamma = gamma  # Proportion of committee members among total users
        self.max_proposals = max_proposals
        self.max_votes = max_votes
        self.max_stake_pool = max_stake_pool

        # Define the rewards for different actions
        self.base_reward_factor = 64
        self.base_rewards_per_epoch = 4
        self.total_balance = 1e3
        self.proposer_chance = 1e-3
        self.commitee_chance = 1e-1

        self.propose_reward = 1  # Placeholder, set according to Ethereum 2.0 specifics
        self.attest_reward = 7 / 8  # Placeholder, set according to Ethereum 2.0 specifics
        self.vote_reward = 6.75 / 8  # Placeholder, set according to Ethereum 2.0 specifics

        self.User = self.create_int_enum('User', ['Proposer', 'Committee', 'Validator'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Attest', 'Propose', 'Vote', 'Wait', 'Stake'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_proposals}, {self.max_votes}, {self.max_stake_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (
            self.alpha, self.gamma, self.max_proposals, self.max_votes, self.max_stake_pool)

    def get_state_space(self) -> Space:
        elements = [(0, self.max_proposals), (0, self.max_votes), self.User, (0, self.max_stake_pool)]
        print(f"Elements for state space: {elements}")
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        action_space = MultiDimensionalDiscreteSpace(self.Action, (0, self.max_proposals))
        print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.User.Validator, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.User.Validator, -1

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[int, int, Enum, int]:
        proposals = state[0]
        votes = state[1]
        user_role = state[2]
        stake_pool = state[3]
        return proposals, votes, user_role, stake_pool

    def get_reward(self, state: BlockchainModel.State, reward_factor: float) -> float:
        proposals, votes, user_role, stake_pool = self.dissect_state(state)
        base_reward = ((stake_pool * self.base_reward_factor) / (self.base_rewards_per_epoch * math.sqrt(self.total_balance)))
        return base_reward * reward_factor

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        proposals, votes, user_role, stake_pool = self.dissect_state(state)
        action_type, action_param = action

        next_user_role = self.User.Validator
        if random() < self.proposer_chance:
            next_user_role = self.User.Proposer
        elif random() < self.commitee_chance:
            next_user_role = self.User.Committee

        if action_type is self.Action.Stake and stake_pool < self.max_stake_pool:
            next_state = (proposals, votes, user_role, stake_pool + 1)
            self.total_balance += 1
            transitions.add(next_state, probability=1, reward=-1)

        elif action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
        elif action_type is self.Action.Attest:
            if user_role in [self.User.Committee,
                             self.User.Validator] and votes < self.max_votes and stake_pool < self.max_stake_pool:
                next_state = (proposals, votes + 1, next_user_role, stake_pool + 1)
                reward = self.get_reward(next_state, self.attest_reward)
                transitions.add(next_state, probability=1, reward=reward)
            else:
                transitions.add(self.final_state, probability=1, reward=0)
        elif action_type is self.Action.Propose:
            if user_role == self.User.Proposer and proposals < self.max_proposals and stake_pool > 0:
                next_state = (proposals + 1, votes, next_user_role, stake_pool - 1)
                reward = self.get_reward(next_state, self.propose_reward)
                transitions.add(next_state, probability=1, reward=reward)
            else:
                transitions.add(self.final_state, probability=1, reward=0)
        elif action_type is self.Action.Vote:
            if (user_role in [self.User.Committee,
                              self.User.Validator] and action_param <= proposals and votes < self.max_votes and
                    stake_pool < self.max_stake_pool):
                next_state = (proposals, votes + 1, next_user_role, stake_pool + 1)
                reward = self.get_reward(next_state, self.vote_reward)
                transitions.add(next_state, probability=1, reward=reward)
            else:
                transitions.add(self.final_state, probability=1, reward=0)
        elif action_type is self.Action.Wait:
            next_state = state
            transitions.add(next_state, probability=1, reward=0)  # No reward for waiting

        # Ensure total transition probabilities sum to 1
        total_prob = sum(transitions.probabilities.values())
        if total_prob == 0:
            num_transitions = len(transitions.probabilities)
            if num_transitions == 0:
                transitions.add((0, 0, self.User.Validator, 0), probability=1)
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
                        remaining_prob = 0
                total_prob = sum(transitions.probabilities.values())
                if not math.isclose(total_prob, 1, abs_tol=1e-5):
                    raise ValueError("Total transition probability does not sum to 1 after adjustment")

        min_difficulty = 1e-5
        for key in transitions.difficulty_contributions.keys():
            if transitions.difficulty_contributions[key] == 0:
                transitions.difficulty_contributions[key] = min_difficulty

        return transitions

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        proposals, votes, user_role, stake_pool = self.dissect_state(state)
        return proposals >= 0 and votes >= 0 and user_role in self.User and stake_pool >= 0

    def get_honest_revenue(self) -> float:
        return self.alpha

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            proposals, votes, user_role, stake_pool = self.state_space.index_to_element(i)
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
            proposals, votes, user_role, stake_pool = self.state_space.index_to_element(i)
            if proposals > votes + 1:
                action = self.Action.Propose
            elif (votes == proposals - 1 and proposals >= 2) or proposals == 0:
                action = self.Action.Illegal
            else:
                action = self.Action.Wait

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('ethereum_model module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = Ethereum2Model(0.35, 0.2, 10, 10, 10)
    print(mdp.get_state_space().size)
