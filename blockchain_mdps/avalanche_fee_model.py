import sys
from typing import Tuple
import numpy as np

from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class EthereumPosModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, clock: int, max_forks: int, fee: float, pool: float, transaction_chance: float):
        self.alpha = alpha  # Honest validators' proportion
        self.gamma = gamma  # Attacker's proportion
        self.max_forks = max_forks  # Maximum number of forks per user
        self.max_sys_forks = int(max_forks * 1e1)
        self.fee = fee  # Tax of block reward
        self.transaction_chance = transaction_chance
        self.pool = pool  # Total pool of rewards
        self.clock = clock

        self.block_reward = 1

        self.Action = self.create_int_enum('Action', ['Illegal', 'AddBlock', 'CreateFork'])

        super().__init__()

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}'
                f'({self.alpha}, {self.gamma}, {self.max_forks}, {self.fee}, {self.pool}, {self.clock}, '
                f'{self.transaction_chance})')

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_forks, self.fee, self.pool, self.clock,
                                self.transaction_chance)

    def get_state_space(self) -> Space:
        """
        状态空间定义为 (num_forks, sys_num_forks, clock)
        - num_forks: 用户创建的分叉数
        - sys_num_forks: 系统分叉数
        - clock: 剩余时间（安全奖励池内的时间单位）
        """
        underlying_space = MultiDimensionalDiscreteSpace(
            (0, int(self.max_forks)),  # 用户分叉数
            (0, self.max_sys_forks),   # 系统分叉数
            (0, int(self.clock))       # 安全奖励池的最大值
        )
        # print(f"Underlying space size: {underlying_space.size}")
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        """
        动作空间定义：
        - Illegal: 非法行为
        - AddBlock: 向主链添加区块
        - CreateFork: 创建分叉
        """
        action_space = DiscreteSpace(self.Action)
        # print(f"Action space size: {action_space.size}")
        return action_space

    def get_initial_state(self) -> BlockchainModel.State:
        """
        初始化状态：主链深度为0，活跃链为主链，没有分叉，奖励池为clock。
        """
        return 0, 0, self.clock

    def get_final_state(self) -> BlockchainModel.State:
        """
        终止状态定义为 (-1, -1, -1)，表示系统结束。
        """
        return -1, -1, -1

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        num_forks, sys_num_forks, clock = state

        # 权重代表用户分叉链成功加入主链的可能性
        weight = num_forks / (sys_num_forks + 1)

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if clock == 0:
            transitions.add((0, 0, self.clock), probability=1, reward=0)
            return transitions

        if action is self.Action.AddBlock:
            reward = self.block_reward * (1 + self.fee * self.transaction_chance)

            if num_forks >= self.max_forks or num_forks < 0:
                transitions.add(self.final_state, probability=1, reward=0)
            else:
                transitions.add((num_forks, sys_num_forks, clock - 1),
                                probability=weight,
                                reward=reward)
                num_forks = min(num_forks + 1, self.max_forks)
                transitions.add((num_forks, min(sys_num_forks + 1, self.max_sys_forks), clock - 1),
                                probability=1 - weight,
                                reward=reward)

        elif action is self.Action.CreateFork:
            reward = max(self.pool * (sys_num_forks - num_forks) / (sys_num_forks + 1), 0)
            self.pool -= reward

            sys_num_forks = min(sys_num_forks + 1, self.max_sys_forks)
            if 0 <= num_forks < self.max_forks:
                transitions.add((num_forks + 1, sys_num_forks, clock - 1),
                                probability=1,
                                reward=reward + 1 - self.fee)
            else:
                transitions.add(self.final_state, probability=1, reward=0)

        return transitions

    def get_honest_revenue(self) -> float:
        """
        获取诚实节点的收益率
        """
        return self.alpha * self.block_reward * (1 + self.fee)

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        """
        判断策略是否是诚实的，默认从状态 (0, 0, 0)
        """
        initial_state = (0, 0, 0)
        return policy[self.state_space.element_to_index(initial_state)] == self.Action.AddBlock
