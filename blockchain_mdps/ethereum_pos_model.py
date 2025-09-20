import logging
import math
import random
from typing import Tuple
from enum import Enum

from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.state_transitions import StateTransitions
from blockchain_mdps.base.blockchain_model import BlockchainModel


class EthereumPoSModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int,
                 fee: float, transaction_chance: float, max_pool: int,
                 max_withhold: int = 8, slashing_cost: float = 10.0):
        # 基本参数（保留）
        self.alpha = alpha  # 初始恶意验证者比例（可在环境中动态改变）
        self.gamma = gamma  # 基本传播延迟概率（基线）
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance
        self.max_pool = max(max_pool, max_fork)

        self.block_reward = 1.0
        self.equivocate_count = 0

        # 新增参数
        self.max_withhold = max_withhold   # withheld pool 的上限（防止无限累积）
        self.slashing_cost = slashing_cost # equivocation 被罚金

        # 枚举：扩展动作集合（保留旧动作语义）
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', [
            'Illegal',           # 保留
            'Withhold',          # 保留（将新块放入 withheld pool）
            'ReleaseOne',        # 释放少量（1）
            'ReleaseAll',        # 释放全部（雪崩场景）
            'Equivocate',        # 双签
            'Attest',              # 投票 / attest
            'Censor',            # 提议时审查（移除交易）
            'SelectiveRelay',    # 选择性转发（部分节点可见）
            'Exit'               # 自愿退出（改变 active stake）
        ])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def get_state_space(self):
        # 原有 chain a/h 各 max_fork 个 (Block, Transaction) 加入额外统计量（放在 tuple 末端）： fork_flag, pool, len_a, len_h, txs_a,
        # txs_h, withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_times100
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork,
                   (0, self.max_pool), (0, self.max_fork), (0, self.max_fork),
                   (0, self.max_pool), (0, self.max_pool),
                   (0, self.max_withhold), (0, self.max_withhold), # withheld counts
                   (0, 1), # partition flag (0/1)
                   (0, self.max_pool), # censor budget (简化)
                   (0, 100)] # active alpha * 100 as int (便于在 state tuple 中代表 stake 分布)
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self):
        # 初始时 withheld 置0, partition_flag 0, censor_budget 0, active_alpha*100 = int(self.alpha*100)
        return self.create_empty_chain() * 2 + (self.Fork.Relevant,) + (0,) * 5 + (0, 0, 0, 0, int(self.alpha * 100))

    def get_final_state(self):
        # final 保持一致但所有末端字段设为 -1，与你先前风格一致
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,) * 10

    def dissect_state(self, state: BlockchainModel.State):
        # 依据上面顺序解包（注意末端字段数量）
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]
        # the last 11 entries (Fork + 10 stats)
        fork = state[-11]
        pool, length_a, length_h, transactions_a, transactions_h, \
            withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100 = state[-10:]
        active_alpha = active_alpha_100 / 100.0
        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h, \
               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha

    # 其余辅助函数（create_empty_chain, is_chain_valid, add_block, shift_back, truncate_chain, chain_length,
    # chain_transactions） 可直接复用你原来的实现（这里略），仍然保持相同语义

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()
        final_state = self.final_state
        EPS = 0.02  # ε扰动比例
        # err = self.error_penalty if hasattr(self, 'error_penalty') else -100
        err = -100

        if state == final_state:
            transitions.add(final_state, probability=1.0, reward=0.0)
            return transitions

        a_len, h_len, fork, w_a, w_h, visibility, pool, active_stake = self.dissect_state(state)

        total_prob = 0.0

        if action == self.Action.Illegal:
            transitions.add(final_state, probability=1.0, reward=err)
            total_prob = 1.0

        elif action == self.Action.Exit:
            transitions.add(final_state, probability=1.0, reward=-0.1)
            total_prob = 1.0

        elif action == self.Action.Withhold:
            if a_len < self.max_fork:
                next_attacker = self.reassemble_state(a_len + 1, h_len, fork, w_a + 1, w_h, visibility, pool,
                                                      active_stake)
                transitions.add(next_attacker, probability=self.alpha, reward=0.1)
                total_prob += self.alpha
            if h_len < self.max_fork:
                next_honest = self.reassemble_state(a_len, h_len + 1, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_honest, probability=1 - self.alpha, reward=0.0)
                total_prob += (1 - self.alpha)

        elif action == self.Action.ReleaseOne:
            if a_len > h_len:
                new_a = max(a_len - 1, 0)
                next_state = self.reassemble_state(new_a, h_len, fork, w_a - 1, w_h, visibility, pool, active_stake)
                transitions.add(next_state, probability=1.0, reward=1.0)
                total_prob = 1.0
            else:
                transitions.add(final_state, probability=1.0, reward=err)
                total_prob = 1.0

        elif action == self.Action.ReleaseAll:
            if a_len > h_len:
                next_state = self.reassemble_state(0, h_len, fork, 0, w_h, visibility, pool, active_stake)
                reward = a_len * 1.0
                transitions.add(next_state, probability=1.0, reward=reward)
                total_prob = 1.0
            else:
                transitions.add(final_state, probability=1.0, reward=err)
                total_prob = 1.0

        elif action == self.Action.Equivocate:
            if a_len + h_len > 0:
                success_prob = a_len / (a_len + h_len)
                next_attack = self.reassemble_state(a_len, 0, fork, w_a, 0, visibility, pool, active_stake)
                next_honest = self.reassemble_state(0, h_len, fork, 0, w_h, visibility, pool, active_stake)
                transitions.add(next_attack, probability=success_prob, reward=0.5)
                transitions.add(next_honest, probability=1 - success_prob, reward=0.0)
                total_prob = 1.0
            else:
                transitions.add(final_state, probability=1.0, reward=err)
                total_prob = 1.0

        elif action == self.Action.Attest:
            if a_len < self.max_fork:
                next_attack = self.reassemble_state(a_len + 1, h_len, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_attack, probability=self.alpha, reward=0.2)
                total_prob += self.alpha
            if h_len < self.max_fork:
                next_honest = self.reassemble_state(a_len, h_len + 1, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_honest, probability=1 - self.alpha, reward=0.0)
                total_prob += (1 - self.alpha)

        elif action == self.Action.SelectiveRelay:
            transitions.add(state, probability=1.0 - EPS, reward=0.05)
            total_prob += (1.0 - EPS)
            if a_len < self.max_fork:
                next_attack = self.reassemble_state(a_len + 1, h_len, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_attack, probability=EPS * self.alpha, reward=0.0)
                total_prob += EPS * self.alpha
            if h_len < self.max_fork:
                next_honest = self.reassemble_state(a_len, h_len + 1, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_honest, probability=EPS * (1 - self.alpha), reward=0.0)
                total_prob += EPS * (1 - self.alpha)

        elif action == self.Action.Censor:
            transitions.add(state, probability=1.0 - EPS, reward=0.05)
            total_prob += (1.0 - EPS)
            if a_len < self.max_fork:
                next_attack = self.reassemble_state(a_len + 1, h_len, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_attack, probability=EPS * self.alpha, reward=0.0)
                total_prob += EPS * self.alpha
            if h_len < self.max_fork:
                next_honest = self.reassemble_state(a_len, h_len + 1, fork, w_a, w_h, visibility, pool, active_stake)
                transitions.add(next_honest, probability=EPS * (1 - self.alpha), reward=0.0)
                total_prob += EPS * (1 - self.alpha)

        if total_prob < 1.0 - 1e-9:
            transitions.add(final_state, probability=1.0 - total_prob, reward=0.0)

        return transitions

    # 你可以保留 get_honest_revenue（或改名）以配合训练监控
    def get_honest_revenue(self) -> float:
        return self.alpha * (1 + self.transaction_chance)
