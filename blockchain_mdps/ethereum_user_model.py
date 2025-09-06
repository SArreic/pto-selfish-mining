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
        self.alpha = alpha  # 攻击者控制的权益比例
        self.gamma = gamma  # 延迟/选择性传播概率
        self.max_fork = max_fork

        # 动作定义（更新）
        self.Action = self.create_int_enum('Action', [
            'Illegal',           # 非法行为
            'Withhold',          # 隐匿区块
            'ReleaseOne',        # 释放一个隐匿区块
            'ReleaseAll',        # 释放全部隐匿区块
            'Equivocate',        # 双签
            'Attest',            # 投票 / attest
            'Censor',            # 审查交易
            'SelectiveRelay',    # 选择性转发
            'Exit'               # 自愿退出
        ])

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork})'

    def get_action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.Action)

    def get_state_space(self) -> DefaultValueSpace:
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork),
                                                         (0, self.max_fork),
                                                         self.Fork)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Irrelevant

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Irrelevant

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork = state

        # 1. 非法行为
        if action == self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 2. 隐匿区块
        elif action == self.Action.Withhold:
            if a >= self.max_fork or h >= self.max_fork:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty/2)
            else:
                transitions.add((a+1, h, self.Fork.Irrelevant), probability=self.alpha)
                transitions.add((a, h+1, self.Fork.Relevant), probability=1-self.alpha)

        # 3. 释放一个隐匿区块
        elif action == self.Action.ReleaseOne:
            if a > 0:
                transitions.add((a-1, h, self.Fork.Relevant), probability=1, reward=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 4. 释放全部隐匿区块
        elif action == self.Action.ReleaseAll:
            if a > h:
                reward = a - h
                transitions.add((0, 0, self.Fork.Irrelevant), probability=1, reward=reward)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 5. 双签
        elif action == self.Action.Equivocate:
            if fork == self.Fork.Relevant and a + h > 0:
                p_success = a / (a + h)
                transitions.add((a, 0, self.Fork.Active), probability=p_success, reward=0)
                transitions.add((0, h, self.Fork.Active), probability=1-p_success, reward=0)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 6. 投票 / attest
        elif action == self.Action.Attest:
            if fork != self.Fork.Active and a < self.max_fork and h < self.max_fork:
                transitions.add((a+1, h, self.Fork.Irrelevant), probability=self.alpha)
                transitions.add((a, h+1, self.Fork.Relevant), probability=1-self.alpha)
            elif fork == self.Fork.Active and 0 < h <= a < self.max_fork:
                transitions.add((a+1, h, self.Fork.Active), probability=self.alpha)
                transitions.add((a-h, 1, self.Fork.Relevant), probability=self.gamma*(1-self.alpha), reward=h)
                transitions.add((a, h+1, self.Fork.Relevant), probability=(1-self.gamma)*(1-self.alpha))
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        # 7. 审查交易
        elif action == self.Action.Censor:
            # 不改变链长，但可能获得小额奖励
            transitions.add((a, h, fork), probability=1, reward=0.1)

        # 8. 选择性转发
        elif action == self.Action.SelectiveRelay:
            # 模拟改变传播延迟（gamma）
            new_gamma = min(1.0, self.gamma + 0.1)
            self.gamma = new_gamma
            transitions.add((a, h, fork), probability=1, reward=0.05)

        # 9. 自愿退出
        elif action == self.Action.Exit:
            # 模拟权益比例下降
            self.alpha = max(0.0, self.alpha - 0.05)
            transitions.add((a, h, fork), probability=1, reward=-0.1)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha

    def build_attack_policy(self) -> BlockchainModel.Policy:
        policy = np.zeros(self.state_space.size, dtype=int)

        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)
            if h > a:
                action = self.Action.Withhold
            elif a >= self.max_fork:
                action = self.Action.ReleaseAll
            elif fork == self.Fork.Relevant and a == h:
                action = self.Action.Equivocate
            else:
                action = self.Action.Attest
            policy[i] = action

        return tuple(policy)

