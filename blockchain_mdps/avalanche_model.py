import sys
from typing import Tuple
import numpy as np

from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class AvalancheAttackModel(BlockchainModel):
    def __init__(self, alpha: float, beta: float, max_forks: int, max_depth: int):
        self.max_depth = max_depth
        self.alpha = alpha  # Honest validators proportion
        self.beta = beta  # Attacker proportion
        self.max_forks = max_forks  # Maximum number of fork chains

        self.Action = self.create_int_enum('Action', ['AddBlock', 'CreateFork', 'SwitchChain', 'PublishFork'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.beta}, {self.max_forks})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.beta, self.max_forks)

    def get_state_space(self) -> Space:
        # State: (main_chain_depth, active_chain_index, num_forks)
        underlying_space = MultiDimensionalDiscreteSpace(
            (0, self.max_depth),  # Main chain depth
            (0, self.max_depth),  # Honest chain depth
            (0, self.max_forks)  # Number of forks
        )
        print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        action_space = DiscreteSpace(self.Action)
        print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        # Start with main chain depth of 0, active chain is main, no forks
        return 0, 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, -1

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        main_chain_depth, honest_chain_depth, num_forks = state

        if action is self.Action.AddBlock:
            if main_chain_depth >= self.max_depth or honest_chain_depth >= self.max_depth:
                # Chain has reached maximum depth, user must publish
                transitions.add(self.final_state, probability=1, reward=0)
            else:
                # No reward for adding block
                transitions.add(
                    (main_chain_depth + 1, honest_chain_depth + 1, num_forks),
                    probability=1,
                    reward=0  # Set reward to 0
                )

        elif action is self.Action.CreateFork:
            if main_chain_depth >= self.max_depth or honest_chain_depth >= self.max_depth:
                # Chain has reached maximum depth, user must publish
                transitions.add(self.final_state, probability=1, reward=0)
            elif num_forks < self.max_forks:
                # No reward for creating fork
                transitions.add(
                    (main_chain_depth, honest_chain_depth + 1, num_forks + 1),  # Creating a new fork
                    probability=1,
                    reward=0  # Set reward to 0
                )
            else:
                transitions.add(self.final_state, probability=1, reward=0)

        elif action is self.Action.SwitchChain:
            if main_chain_depth <= 0 < num_forks:
                # No reward for switching chain
                transitions.add(
                    (1, honest_chain_depth, num_forks - 1),  # Switching to a longer chain
                    probability=self.beta,
                    reward=0  # Set reward to 0
                )
            else:
                transitions.add(self.final_state, probability=1, reward=0)

        elif action is self.Action.PublishFork:
            attack_weight = (main_chain_depth + 1.5 * num_forks) * self.beta
            honest_weight = honest_chain_depth * self.alpha
            new_length = min(main_chain_depth, honest_chain_depth)

            if attack_weight > honest_weight:
                transitions.add(
                    (main_chain_depth - new_length, honest_chain_depth - new_length, num_forks),
                    probability=1,
                    reward=new_length
                )
            else:
                transitions.add(
                    (main_chain_depth - new_length, honest_chain_depth - new_length, num_forks),
                    probability=1,
                    reward=0
                )
        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        initial_state = (0, 0, 0)
        return policy[self.state_space.element_to_index(initial_state)] == self.Action.AddBlock

    def build_honest_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            if active_chain == 0:
                action = self.Action.AddBlock
            else:
                action = self.Action.SwitchChain

            policy[i] = action

        return tuple(policy)

    def build_attack_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            active_chain, num_forks, total_weight = self.state_space.index_to_element(i)

            if active_chain != 0:
                action = self.Action.SwitchChain
            elif num_forks < self.max_forks:
                action = self.Action.CreateFork
            else:
                action = self.Action.PublishFork

            policy[i] = action

        return tuple(policy)


if __name__ == '__main__':
    print('simplified_avalanche_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = AvalancheAttackModel(0.6, 0.4, 3)
    print(mdp.state_space.size)
    p = mdp.build_attack_policy()
    print(p[:10])
