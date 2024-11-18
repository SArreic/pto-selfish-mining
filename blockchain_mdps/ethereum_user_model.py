import sys
from typing import Tuple
import numpy as np

from . import StateTransitions
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.blockchain_model import BlockchainModel

class EthereumUserModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int):
        self.alpha = alpha  # 攻击者控制的节点比例
        self.gamma = gamma  # 延迟传播概率
        self.max_fork = max_fork

        # 动作定义
        self.Action = self.create_int_enum('Action', ['Illegal', 'Withhold', 'Release', 'Equivocate', 'Vote'])
        # 定义分叉和区块链的状态枚举
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork})'

    def get_action_space(self) -> DiscreteSpace:
        # 动作空间定义
        return DiscreteSpace(self.Action)

    def get_state_space(self) -> DefaultValueSpace:
        # 状态空间：攻击者链长度、诚实节点链长度和分叉状态
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Irrelevant  # 初始状态

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Irrelevant  # 终止状态

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        # 状态转移实现
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork = state

        # 动作：非法操作
        if action == self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        # 动作：隐匿区块
        elif action == self.Action.Withhold:
            if a >= self.max_fork or h >= self.max_fork:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            else:
                next_state = a + 1, h, self.Fork.Irrelevant
                transitions.add(next_state, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant
                transitions.add(honest_block, probability=1 - self.alpha)

        # 动作：发布隐匿区块（雪崩攻击）
        elif action == self.Action.Release:
            if a > h:
                next_state = a - h - 1, 0, self.Fork.Irrelevant
                transitions.add(next_state, probability=1, reward=h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 动作：多重签名投票（平衡攻击）
        elif action == self.Action.Equivocate:
            if fork == self.Fork.Relevant:
                next_state = a, h, self.Fork.Active
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 动作：投票或等待
        elif action == self.Action.Vote:
            if fork != self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_block = a + 1, h, self.Fork.Irrelevant
                transitions.add(attacker_block, probability=self.alpha)

                honest_block = a, h + 1, self.Fork.Relevant
                transitions.add(honest_block, probability=1 - self.alpha)
            elif fork == self.Fork.Active and 0 < h <= a < self.max_fork:
                attacker_block = a + 1, h, self.Fork.Active
                transitions.add(attacker_block, probability=self.alpha)

                honest_support_block = a - h, 1, self.Fork.Relevant
                transitions.add(honest_support_block, probability=self.gamma * (1 - self.alpha), reward=h)

                honest_adversary_block = a, h + 1, self.Fork.Relevant
                transitions.add(honest_adversary_block, probability=(1 - self.gamma) * (1 - self.alpha))
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def build_attack_policy(self) -> BlockchainModel.Policy:
        # 基于攻击者视角的策略生成
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)

            if h > a:
                action = self.Action.Withhold
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Release
            elif (h == 1 and a == 1) and fork == self.Fork.Relevant:
                action = self.Action.Equivocate
            else:
                action = self.Action.Vote

            policy[i] = action

        return tuple(policy)
