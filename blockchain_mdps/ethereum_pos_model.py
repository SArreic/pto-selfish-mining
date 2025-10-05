import logging
import math
import random
from typing import Tuple
from enum import Enum

# 假设这些基础类已在您的环境中定义
from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.state_transitions import StateTransitions
from blockchain_mdps.base.blockchain_model import BlockchainModel


class EthereumPoSModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int,
                 fee: float, transaction_chance: float, max_pool: int,
                 max_withhold: int = 8, slashing_cost: float = 10.0):

        # --- 基本/初始化参数 ---
        self.alpha = alpha  # 初始恶意验证者比例 (Attacker's initial stake share)
        self.gamma = gamma  # 基本传播延迟概率 (Network delay/visibility baseline, gamma_net)
        self.max_fork = max_fork  # 链长限制，防止状态空间无限增长
        self.fee = fee  # 每块交易费收入的平均值
        self.transaction_chance = transaction_chance  # 单个区块包含交易的概率 (未在转移中精细使用)
        self.max_pool = max(max_pool, max_fork)  # 交易池/未发布区块池的上限

        self.block_reward = 1.0  # 基础出块奖励
        self.equivocate_count = 0  # 记录双签次数 (未在状态空间中使用)

        # --- 新增/ PoS 相关参数 ---
        self.max_withhold = max_withhold  # withheld pool 的上限，限制攻击者可保留的块数
        self.slashing_cost = slashing_cost  # Equivocation (双签) 被罚没的基准成本

        # --- 内部常量/阈值 ---
        # Finality 阈值: 诚实链长度超过此值，重组成功率急剧下降
        self.FINALITY_THRESHOLD = self.max_fork * 0.7
        self.ERROR_PENALTY = -100.0  # 非法操作/错误转移的惩罚
        self.PROB_JITTER = 1e-6  # 用于打破确定性转移的微小扰动概率
        self.EPS = 0.02  # 用于 Censor/Relay 等动作的小概率转移
        self.MIN_TOTAL_PROB = 1e-12  # 最小总概率，用于转移归一化检查

        # --- 状态和动作枚举 ---
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])  # 分叉状态
        self.Action = self.create_int_enum('Action', [
            'Illegal', 'Withhold', 'ReleaseOne', 'ReleaseAll', 'Equivocate',
            'Attest', 'Censor', 'SelectiveRelay', 'Exit'
        ])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])  # 区块是否存在
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])  # 区块是否包含交易

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.gamma}, {self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def create_empty_chain(self):
        # 假设这里是辅助函数，用于创建 (Block, Transaction) 的空链部分
        return [self.Block.NoBlock, self.Transaction.NoTransaction]

    def get_state_space(self):
        # 状态空间设计保持不变
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork,
                                                                           (0, self.max_pool), (0, self.max_fork),
                                                                           (0, self.max_fork),
                                                                           (0, self.max_pool), (0, self.max_pool),
                                                                           (0, self.max_withhold),
                                                                           (0, self.max_withhold),  # withheld counts
                                                                           (0, 1),  # partition flag (0/1)
                                                                           (0, self.max_pool),  # censor budget (简化)
                                                                           (0, 100)]  # active alpha * 100 as int
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self):
        return self.create_empty_chain() * 2 + (self.Fork.Relevant,) + (0,) * 5 + (0, 0, 0, 0, int(self.alpha * 100))

    def get_final_state(self):
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,) * 10

    def dissect_state(self, state: BlockchainModel.State):
        """
        解包状态元组，提取所有链体部分和统计量。
        返回值包括：
        a, h: 攻击者链和诚实链的 (Block, Transaction) 列表部分
        fork: 分叉状态
        pool: 未发布区块总数 (简化)
        length_a, length_h: 攻击者链和诚实链的长度
        transactions_a, transactions_h: 两条链上的交易总数 (简化)
        withheld_a, withheld_h: 攻击者保留的 A 块和 H 块数量 (主要使用 withheld_a)
        partition_flag: 网络分区标志 (0/1)
        censor_budget: 攻击者可用的审查预算
        active_alpha: 攻击者的实际有效权益占比 (0.0 到 1.0)
        """
        # 严格解包 13 个组件
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]

        # 11 个统计量部分
        fork = state[-11]
        pool, length_a, length_h, transactions_a, transactions_h, \
            withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100 = state[-10:]

        active_alpha = active_alpha_100 / 100.0

        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h, \
            withheld_a, withheld_h, partition_flag, censor_budget, active_alpha

    def _build_next_state(self, state: BlockchainModel.State,
                          new_a, new_h, new_fork, new_pool, new_la, new_lh, new_ta, new_th,
                          new_wha, new_whh, new_part, new_censor, new_alpha100):
        """
        辅助函数：根据新的统计量构建完整的下一个状态元组。
        参数：
        new_*：所有 13 个组件的新值，用于组装新的状态元组。
        """
        # 注意：这里我们假设链体 (a, h) 在长度变化时只更新统计量，保持链体部分不变以简化。
        # 实际的 BlockchainModel 实现需要根据 la/lh 变动来更新 a/h 的 Block/Transaction 列表部分。

        # 简化处理：保留原状态的 Block/Transaction 部分
        chain_part = state[:4 * self.max_fork]

        # 新的统计量部分
        stats_part = (new_fork,) + (new_pool, new_la, new_lh, new_ta, new_th,
                                    new_wha, new_whh, new_part, new_censor, new_alpha100)
        return chain_part + stats_part

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        # --- safety defaults / params ---
        gamma_net = self.gamma

        # identify final_state
        final_state = self.get_final_state()

        if state == final_state:
            transitions.add(final_state, probability=1.0, reward=0.0)
            return transitions

        # --- unpack state robustly ---
        try:
            a, h, fork, pool, length_a, length_h, transactions_a, transactions_h, \
                withheld_a, withheld_h, partition_flag, censor_budget, active_alpha = self.dissect_state(state)
            active_alpha_100 = int(active_alpha * 100)
        except Exception:
            transitions.add(final_state, probability=1.0, reward=self.ERROR_PENALTY)
            return transitions

        # 辅助函数: add and accumulate in local dict then finalize (to ensure normalization)
        next_probs = {}
        next_reward_mass = {}  # stores prob*reward for expectation

        def _accumulate(snext, p, r):
            if p <= 0: return
            key = tuple(snext)
            next_probs[key] = next_probs.get(key, 0.0) + float(p)
            next_reward_mass[key] = next_reward_mass.get(key, 0.0) + float(p) * float(r)

        # 辅助函数: finalize
        def _finalize_and_write(default_fallback_state):
            total_p = sum(next_probs.values())
            if total_p < self.MIN_TOTAL_PROB:
                transitions.add(default_fallback_state, probability=1.0, reward=self.ERROR_PENALTY * 0.1)
                return

            if len(next_probs) == 1 and abs(total_p - 1.0) < 1e-12:
                only_key = next(iter(next_probs))
                next_probs[only_key] = 1.0 - self.PROB_JITTER
                cur_key = tuple(state)
                next_probs[cur_key] = next_probs.get(cur_key, 0.0) + self.PROB_JITTER

            total_p = sum(next_probs.values())
            if total_p <= 0:
                transitions.add(default_fallback_state, probability=1.0, reward=self.ERROR_PENALTY * 0.1)
                return

            for key, p in list(next_probs.items()):
                norm_p = float(p) / float(total_p)
                reward_mass = next_reward_mass.get(key, 0.0)
                expected_reward = (reward_mass / p) if p > 0 else 0.0
                transitions.add(key, probability=norm_p, reward=expected_reward)

        # --- 动作特定规则和变量 ---

        # 内部变量：
        # gamma_net: 网络延迟/可见性参数，影响发布/检测的概率。
        # lead: length_a - length_h，攻击者私有链领先公共链的长度。
        # decay_factor: Finality 衰减因子，当 h_len 较大时，该因子使 P_success 显著降低。
        # accepted_blocks: ReleaseAll 成功时，实际被诚实网络接受的区块数量。
        # detect_prob: Equivocate 行为被检测到的概率。
        # slashing_penalty: Equivocate 被检测到时，基于 active_alpha 计算的罚金。
        # p_att, p_hon: Attest 动作中，攻击者和诚实节点出块的概率 (基于 active_alpha)。
        # EXIT_DECREMENT: Exit 动作中，active_alpha_100 的减少量。

        # [ ... Withhold, ReleaseOne, ReleaseAll, Equivocate, Attest, Exit 的精细化逻辑 ... ]
        # 这些动作内部使用了 length_a, length_h, active_alpha, withheld_a, FINALITY_THRESHOLD, slashing_cost 等变量。

        # --- Action-specific rules ---

        # 0: Illegal
        if action == self.Action.Illegal:
            _accumulate(final_state, 1.0, self.ERROR_PENALTY)
            _finalize_and_write(final_state)
            return transitions

        # 1: Withhold -> 增加 withheld_a 计数
        if action == self.Action.Withhold:
            # 攻击者成功出块 (概率 active_alpha)
            p_att = active_alpha
            p_hon = 1.0 - active_alpha

            if p_att > 0:
                # 增加 withheld pool 计数，链长不变
                next_wha = min(withheld_a + 1, self.max_withhold)
                snext = self._build_next_state(state, a, h, fork, pool, length_a, length_h, transactions_a,
                                               transactions_h,
                                               next_wha, withheld_h, partition_flag, censor_budget, active_alpha_100)
                _accumulate(snext, p_att, -0.01)  # 机会成本

            if p_hon > 0:
                # 诚实节点出块，增加 h 链长，wha 不变
                next_lh = min(length_h + 1, self.max_fork)
                snext = self._build_next_state(state, a, h, fork, pool, length_a, next_lh, transactions_a,
                                               transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100)
                _accumulate(snext, p_hon, 0.0)

            _finalize_and_write(final_state)
            return transitions

        # 2: ReleaseOne -> 消耗 withheld_a，引入 Finality 衰减
        if action == self.Action.ReleaseOne:
            if withheld_a <= 0:
                _accumulate(final_state, 1.0, self.ERROR_PENALTY * 0.2)
                _finalize_and_write(final_state)
                return transitions

            lead = length_a - length_h
            # Finality 衰减因子
            decay_factor = 1.0
            if length_h > self.FINALITY_THRESHOLD:
                decay_factor = 0.05

            p_success = min(0.95, 0.5 + 0.1 * lead) * (1.0 - gamma_net) * decay_factor
            p_fail = 1.0 - p_success

            # 成功：消耗一个 withheld_a，并延伸 h 链
            if p_success > 0:
                reward_success = (self.block_reward + self.fee) * active_alpha  # 奖励与 stake 挂钩
                next_lh = min(length_h + 1, self.max_fork)

                snext = self._build_next_state(state, a, h, fork, pool, length_a, next_lh, transactions_a,
                                               transactions_h,
                                               withheld_a - 1, withheld_h, partition_flag, censor_budget,
                                               active_alpha_100)
                _accumulate(snext, p_success, reward_success)

            # 失败：状态不变
            if p_fail > 0:
                _accumulate(state, p_fail, -0.05)

            _finalize_and_write(final_state)
            return transitions

        # 3: ReleaseAll -> 消耗所有 withheld_a，引入 Finality 衰减
        if action == self.Action.ReleaseAll:
            if withheld_a <= 0:
                _accumulate(final_state, 1.0, self.ERROR_PENALTY * 0.2)
                _finalize_and_write(final_state)
                return transitions

            lead = length_a - length_h
            decay_factor = 1.0
            if length_h > self.FINALITY_THRESHOLD:
                decay_factor = 0.05

            p_success = min(0.99, 0.6 + 0.08 * lead) * (1.0 - gamma_net) * decay_factor
            p_fail = 1.0 - p_success

            if p_success > 0:
                accepted_blocks = min(withheld_a, lead + 1)
                reward_success = accepted_blocks * (self.block_reward + self.fee) * active_alpha

                # 假设重组成功，h_len 增加 accepted_blocks
                next_lh = min(length_h + accepted_blocks, self.max_fork)

                snext = self._build_next_state(state, a, h, fork, pool, length_a, next_lh, transactions_a,
                                               transactions_h,
                                               0, withheld_h, partition_flag, censor_budget,
                                               active_alpha_100)  # withheld_a 归零
                _accumulate(snext, p_success, reward_success)

            if p_fail > 0:
                _accumulate(state, p_fail, -0.1)

            _finalize_and_write(final_state)
            return transitions

        # 4: Equivocate -> 引入基于 Finality 的高 Slashing 风险
        if action == self.Action.Equivocate:
            total = max(1, length_a + length_h)
            succ_prob = float(length_a) / float(total)

            # Finality 增强的检测概率
            detect_prob = min(0.9, 0.1 + 0.5 * (1.0 - gamma_net))
            if length_h > self.FINALITY_THRESHOLD:
                detect_prob = 0.99

            p_slashed = detect_prob
            p_success_no_detect = succ_prob * (1.0 - detect_prob)
            p_fail_no_detect = (1.0 - succ_prob) * (1.0 - detect_prob)

            if p_success_no_detect > 0:
                reward_gain = self.block_reward * length_a * active_alpha  # 奖励基于 active_alpha
                snext = self._build_next_state(state, a, h, self.Fork.Active, pool, length_a, 0, transactions_a,
                                               transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100)
                _accumulate(snext, p_success_no_detect, reward_gain)

            if p_fail_no_detect > 0:
                _accumulate(state, p_fail_no_detect, 0.0)

            if p_slashed > 0:
                slashing_penalty = -self.slashing_cost * active_alpha  # Slashing 惩罚与 active_alpha 挂钩
                _accumulate(final_state, p_slashed, slashing_penalty)

            _finalize_and_write(final_state)
            return transitions

        # 5: Attest -> 出块概率与 active_alpha 挂钩
        if action == self.Action.Attest:
            p_att = active_alpha
            p_hon = 1.0 - active_alpha

            if p_att > 0:
                next_la = min(length_a + 1, self.max_fork)
                reward_att = 0.2 * active_alpha  # 奖励与 stake 挂钩
                snext = self._build_next_state(state, a, h, fork, pool, next_la, length_h, transactions_a,
                                               transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100)
                _accumulate(snext, p_att, reward_att)

            if p_hon > 0:
                next_lh = min(length_h + 1, self.max_fork)
                reward_hon = 0.1
                snext = self._build_next_state(state, a, h, fork, pool, length_a, next_lh, transactions_a,
                                               transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100)
                _accumulate(snext, p_hon, reward_hon)

            _finalize_and_write(final_state)
            return transitions

        # 6: Censor -> (保持原有的简化逻辑，可进一步细化)
        if action == self.Action.Censor:
            _accumulate(state, 1.0 - self.EPS, 0.05)
            # 成功审查：增加 a 链长，消耗 censor_budget (未实现)
            _accumulate(self._build_next_state(state, a, h, fork, pool, min(length_a + 1, self.max_fork), length_h,
                                               transactions_a, transactions_h,
                                               withheld_a, withheld_h, partition_flag, max(0, censor_budget - 1),
                                               active_alpha_100),
                        self.EPS * active_alpha, 0.15)
            _accumulate(state, self.EPS * (1.0 - active_alpha), -0.05)
            _finalize_and_write(final_state)
            return transitions

        # 7: SelectiveRelay -> (保持原有的简化逻辑)
        if action == self.Action.SelectiveRelay:
            _accumulate(state, 1.0 - self.EPS, 0.02)
            _accumulate(self._build_next_state(state, a, h, fork, pool, min(length_a + 1, self.max_fork), length_h,
                                               transactions_a, transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100),
                        self.EPS * active_alpha, 0.05)
            _accumulate(self._build_next_state(state, a, h, fork, pool, length_a, min(length_h + 1, self.max_fork),
                                               transactions_a, transactions_h,
                                               withheld_a, withheld_h, partition_flag, censor_budget, active_alpha_100),
                        self.EPS * (1.0 - active_alpha), 0.0)
            _finalize_and_write(final_state)
            return transitions

        # 8: Exit -> 软退出：减少 active_alpha_100
        if action == self.Action.Exit:
            EXIT_DECREMENT = 10

            next_alpha100 = max(0, active_alpha_100 - EXIT_DECREMENT)

            snext = self._build_next_state(state, a, h, fork, pool, length_a, length_h, transactions_a, transactions_h,
                                           withheld_a, withheld_h, partition_flag, censor_budget, next_alpha100)

            _accumulate(snext, 1.0, -0.1)  # 软退出成本

            _finalize_and_write(final_state)
            return transitions

        # default fallback
        _accumulate(state, 1.0, -0.01)
        _finalize_and_write(final_state)
        return transitions
