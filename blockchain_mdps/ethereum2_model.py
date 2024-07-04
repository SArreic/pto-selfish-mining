import math
import sys
from enum import Enum
from random import random
from typing import Tuple

import numpy as np

from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class Ethereum2Model(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_stake_pool: int):
        self.alpha = alpha if alpha is not None else 0.2  # Propose chance
        self.gamma = gamma if gamma is not None else 0.8  # Vote chance
        self.max_stake_pool = max_stake_pool

        # Define the rewards for different actions
        self.base_reward_factor = 6.4  # Scaled down
        self.base_rewards_per_epoch = 0.4  # Scaled down
        self.total_balance = 100.0

        self.propose_reward = 1e2  # Scaled down
        self.vote_reward = (7 / 8) * 1e2  # Scaled down
        self.base_reward = (6.75 / 8) * 1e2  # Scaled down

        self.User = self.create_int_enum('User', ['Proposer', 'Validator', 'Invalid'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Stake', 'Propose', 'Vote', 'Wait'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_stake_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (
            self.alpha, self.gamma, self.max_stake_pool)

    def get_state_space(self) -> Space:
        elements = [(0, 1), (0, 1), self.User, (0, self.max_stake_pool), (0, self.max_stake_pool)]
        print(f"Elements for state space: {elements}")
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        action_space = DiscreteSpace(self.Action)
        print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.User.Validator, 0, 0  # Note the change here: reputation replaced by failures

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.User.Invalid, -1, -1

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[int, int, Enum, int, int]:
        propose_success = state[0]
        vote_success = state[1]
        user_role = state[2]
        stake_pool = state[3]
        failures = state[4]
        return propose_success, vote_success, user_role, stake_pool, failures

    def get_reward(self, state: BlockchainModel.State, reward_factor: float) -> float:
        propose_success, vote_success, user_role, stake_pool, failures = self.dissect_state(state)
        print(f"Calculating reward: state={state}, reward_factor={reward_factor}, total_balance={self.total_balance}")
        reward = (reward_factor * (stake_pool + 1.0)) / self.total_balance
        print(f"Calculated reward: {reward}")
        return reward

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()
        print(f"Current state: {state}, Action: {action}")

        # if not self.is_state_valid(state):
        #     transitions.add(self.final_state, probability=1, reward=0)
        #     return transitions

        if state == self.final_state:
            transitions.add(self.final_state, probability=1, reward=0)
            return transitions

        propose_success, vote_success, user_role, stake_pool, failures = self.dissect_state(state)
        action_type = action

        if failures >= 5:
            stake_pool = max(0, stake_pool - 1)
            failures = 0

        if stake_pool <= 1:
            if stake_pool < 1:
                self.total_balance += 1
                next_state = (propose_success, vote_success, self.User.Validator, stake_pool + 1, failures)
                transitions.add(next_state, probability=1, reward=-1)
                print(f"Next state: {next_state}, Probability: 1, Reward: -1")
            else:
                next_state = (propose_success, vote_success, self.User.Validator, stake_pool, failures)
                transitions.add(next_state, probability=1, reward=0)
                print(f"Next state: {next_state}, Probability: 1, Reward: 0")

            return transitions

        if user_role is self.User.Validator:
            if random() < self.alpha:
                user_role = self.User.Proposer

        if action_type is self.Action.Stake:
            next_stake = stake_pool + 1 if stake_pool < self.max_stake_pool else stake_pool
            next_state = (propose_success, vote_success, user_role, next_stake, failures)
            self.total_balance += 1.0
            transitions.add(next_state, probability=1, reward=-1.0)
            print(f"Next state: {next_state}, Probability: 1, Reward: -1.0")

        elif action_type is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            print(f"Next state: {self.final_state}, Probability: 1, Reward: {self.error_penalty / 2}")

        elif action_type is self.Action.Propose:
            if user_role == self.User.Proposer and stake_pool > 0:
                next_state = (1, vote_success, self.User.Validator, stake_pool, failures)
                reward = self.get_reward(next_state, self.propose_reward)
                transitions.add(next_state, probability=self.alpha, reward=reward, difficulty_contribution=stake_pool)
                print(
                    f"Next state: {next_state}, Probability: {self.alpha}, Reward: {reward}, Difficulty: {stake_pool}")

                next_failures = failures + 1
                reward = self.get_reward(next_state, self.base_reward)
                next_state = (0, vote_success, self.User.Validator, stake_pool, next_failures)
                transitions.add(next_state, probability=1 - self.alpha, reward=reward, difficulty_contribution=stake_pool)
                print(
                    f"Next state: {next_state}, Probability: {1 - self.alpha}, Reward: {reward}, Difficulty: {stake_pool}")
            else:
                transitions.add(state, probability=1, reward=0)
                print(f"Next state: {state}, Probability: 1, Reward: 0")

        elif action_type is self.Action.Vote:
            if user_role == self.User.Validator and stake_pool > 0:
                next_state = (propose_success, 1, user_role, stake_pool, failures)
                reward = self.get_reward(next_state, self.vote_reward)
                transitions.add(next_state, probability=self.gamma, reward=reward, difficulty_contribution=stake_pool)
                print(
                    f"Next state: {next_state}, Probability: {self.gamma}, Reward: {reward}, Difficulty: {stake_pool}")

                next_failures = failures + 1
                reward = self.get_reward(next_state, self.base_reward)
                next_state = (propose_success, 0, user_role, stake_pool, next_failures)
                transitions.add(next_state, probability=1 - self.gamma, reward=reward,
                                difficulty_contribution=stake_pool)
                print(
                    f"Next state: {next_state}, Probability: {1 - self.gamma}, Reward: {reward}, Difficulty: {stake_pool}")
            else:
                transitions.add(state, probability=1, reward=0)
                print(f"Next state: {state}, Probability: 1, Reward: 0")

        elif action_type is self.Action.Wait:
            if user_role == self.User.Invalid:
                next_state = (0, 0, self.User.Proposer, stake_pool, 0)
                transitions.add(next_state, probability=self.alpha, reward=0)
                print(f"Next state: {state}, Probability: {self.alpha}, Reward: 0")

                next_state = (0, 0, self.User.Validator, stake_pool, 0)
                transitions.add(next_state, probability=self.gamma, reward=0)
                print(f"Next state: {state}, Probability: {self.gamma}, Reward: 0")

            elif user_role == self.User.Validator or user_role == self.User.Proposer:
                next_state = (propose_success, vote_success, user_role, stake_pool, failures)
                reward = self.get_reward(next_state, self.base_reward)
                transitions.add(next_state, probability=1, reward=reward,
                                difficulty_contribution=propose_success + vote_success)
                print(
                    f"Next state: {next_state}, Probability: 1, Reward: {reward}, Difficulty: {propose_success + vote_success}")

            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)
                print(
                    f"Next final state: {self.final_state}, Probability: 1, Reward: {self.error_penalty}")

        return transitions

    def is_action_valid(self, state: BlockchainModel.State, action: BlockchainModel.Action) -> bool:
        propose_success, vote_success, user_role, stake_pool, failures = self.dissect_state(state)
        action_type, action_param = action
        if user_role == self.User.Invalid and action_type != self.Action.Stake:
            return False
        if action_type is self.Action.Stake and stake_pool < self.max_stake_pool:
            return True
        if action_type is self.Action.Illegal:
            return True
        if action_type is self.Action.Propose and user_role == self.User.Proposer and stake_pool > 0:
            return True
        if action_type is self.Action.Vote and user_role != self.User.Invalid and stake_pool > 0:
            return True
        if action_type is self.Action.Wait:
            return True
        return False

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        propose_success, vote_success, user_role, stake_pool, failures = self.dissect_state(state)
        return (0 <= propose_success <= 1 and 0 <= vote_success <= 1
                and 0 <= stake_pool <= self.max_stake_pool and 0 <= failures <= 5)

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

    mdp = Ethereum2Model(0.35, 0.2, 10)
    print(mdp.get_state_space().size)
