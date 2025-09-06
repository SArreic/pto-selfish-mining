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
        """
        Robust transition builder:
          - accumulate candidate next states in local dicts (prob, prob-weighted reward)
          - perform sanity fixes: if total_prob == 0 -> add self-loop; if only one deterministic -> add tiny self-loop jitter
          - scale down extreme penalties (self.error_penalty should be reasonably sized)
          - finally add into StateTransitions
        """
        transitions = StateTransitions()

        # safety params (tune if needed)
        PROB_JITTER = 1e-3  # 如果某转移是确定性的，给主转移 1-PROJ 并把 PROJ 给 self-loop
        MIN_TOTAL_PROB = 1e-12  # 近似0 的阈值
        # 确保你的类里有合理的 error_penalty，例如 -100 而非 -100000
        if not hasattr(self, 'error_penalty'):
            self.error_penalty = -100.0

        def _add_local(next_probs: dict, next_rew_acc: dict, s_next, p, r):
            """accumulate probability and reward mass for a candidate next state key (use state tuples as keys)."""
            if p <= 0:
                return
            # ensure key is hashable tuple
            key = tuple(s_next) if isinstance(s_next, (list, tuple)) else s_next
            next_probs[key] = next_probs.get(key, 0.0) + float(p)
            # accumulate reward mass (prob * reward) to compute expected reward later
            next_rew_acc[key] = next_rew_acc.get(key, 0.0) + float(p) * float(r)

        # helpers
        def _finalize_and_write(next_probs: dict, next_rew_acc: dict, current_state):
            """Normalize, add jitter if needed, compute expected rewards and write to transitions."""
            total_p = sum(next_probs.values())
            # case: no outgoing transitions -> add self-loop with small penalty  (avoid zero row)
            if total_p < MIN_TOTAL_PROB:
                # prefer keep-in-place with small negative reward instead of jumping to final_state
                transitions.add(current_state, probability=1.0, reward=-0.01)
                return

            # If deterministic single next state with prob≈1, add tiny self-loop jitter to break perfect determinism
            if len(next_probs) == 1:
                (only_state,), = (list(next_probs.items()),)
                # items() trick above is a little awkward; better use:
                only_key = next(iter(next_probs))
                p0 = next_probs[only_key]
                if abs(p0 - 1.0) < 1e-12:
                    # reduce main prob slightly, give PROB_JITTER to self-loop
                    next_probs[only_key] = 1.0 - PROB_JITTER
                    cur_key = tuple(current_state) if isinstance(current_state, (tuple, list)) else current_state
                    next_probs[cur_key] = next_probs.get(cur_key, 0.0) + PROB_JITTER
                    # distribute reward mass: if self-loop wasn't present, its accumulated reward is currently 0
                    # We leave next_rew_acc for only_key untouched; self-loop reward will be computed below as next_rew_acc.get(self,0)/prob
                    # (we will use 0 reward for self-loop if none accumulated)
                    total_p = 1.0

            # Now normalize probabilities (in case they don't sum exactly to 1)
            total_p = sum(next_probs.values())
            if total_p <= 0:
                transitions.add(current_state, probability=1.0, reward=-0.01)
                return

            for key, p in list(next_probs.items()):
                normalized_p = float(p) / float(total_p)
                # expected reward for this next-state:
                reward_mass = next_rew_acc.get(key, 0.0)
                expected_reward = (reward_mass / p) if p > 0 else 0.0
                # ensure key is the state tuple object expected by transitions.add
                next_state_obj = tuple(key) if isinstance(key, tuple) else key
                transitions.add(next_state_obj, probability=normalized_p, reward=expected_reward)

        # ---- build local candidate dicts instead of adding directly to transitions ----
        next_probs = {}
        next_rew_acc = {}  # stores sum(prob * reward) for each next-state, to compute expected reward
        cur_state = state

        # handle final_state corner
        if check_valid and not self.is_state_valid(state):
            # instead of jumping to final_state with huge penalty, add small negative self-loop to keep matrix well-formed
            _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.01)
            _finalize_and_write(next_probs, next_rew_acc, state)
            return transitions

        if state == self.final_state:
            transitions.add(self.final_state, probability=1.0, reward=0.0)
            return transitions

        # ------------------ original transition logic (use _add_local instead of transitions.add) ------------
        a, h, fork = state

        # Illegal: instead of final_state with huge penalty, penalize but keep self-loop fallback
        if action == self.Action.Illegal:
            # penalize but do not create absorbing sink; prefer a failed self-loop for numerical stability
            _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.1)

        elif action == self.Action.Withhold:
            if a >= self.max_fork or h >= self.max_fork:
                # invalid -> small penalty but keep in state (not absorbing)
                _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.1)
            else:
                # attacker advances with prob alpha
                next_s = (a + 1, h, self.Fork.Irrelevant)
                _add_local(next_probs, next_rew_acc, next_s, self.alpha, 0.0)
                # honest advances with prob 1-alpha
                next_s2 = (a, h + 1, self.Fork.Relevant)
                _add_local(next_probs, next_rew_acc, next_s2, 1.0 - self.alpha, 0.0)

        elif action == self.Action.ReleaseOne:
            if a > 0 and a > h:
                # success likely -> reduce attacker's withheld by one (approx)
                next_s = (a - 1, 0, self.Fork.Irrelevant)
                # reward small and proportional to h+1 (or some smoother function)
                _add_local(next_probs, next_rew_acc, next_s, 1.0, float(h + 1))
            else:
                # failure -> small negative reward but keep state instead of jumping to final_state
                _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.05)

        elif action == self.Action.ReleaseAll:
            # more delicate: if attacker chain dominates, large payoff; otherwise small cost
            if a > h:
                # partial success: collapse to cleaned state
                next_s = (0, 0, self.Fork.Irrelevant)
                _add_local(next_probs, next_rew_acc, next_s, 1.0, float(a - h))
            else:
                # not profitable, small cost but keep state
                _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.05)

        elif action == self.Action.Equivocate:
            if fork == self.Fork.Relevant and (a + h) > 0:
                succ_prob = float(a) / float(a + h)
                next_s1 = (a, 0, self.Fork.Active)
                next_s2 = (0, h, self.Fork.Active)
                _add_local(next_probs, next_rew_acc, next_s1, succ_prob, 0.0)
                _add_local(next_probs, next_rew_acc, next_s2, 1.0 - succ_prob, 0.0)
            else:
                _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.05)

        elif action == self.Action.Attest:
            # Attest/fast path with gamma (propagation) effect
            # If fork inactive
            if fork != self.Fork.Active and a < self.max_fork and h < self.max_fork:
                next_s1 = (a + 1, h, self.Fork.Irrelevant)
                next_s2 = (a, h + 1, self.Fork.Relevant)
                _add_local(next_probs, next_rew_acc, next_s1, self.alpha, 0.0)
                _add_local(next_probs, next_rew_acc, next_s2, 1.0 - self.alpha, 0.0)
            elif fork == self.Fork.Active and (0 < h <= a < self.max_fork):
                next_s1 = (a + 1, h, self.Fork.Active)
                _add_local(next_probs, next_rew_acc, next_s1, self.alpha, 0.0)
                next_s2 = (a - h, 1, self.Fork.Relevant)
                next_s3 = (a, h + 1, self.Fork.Relevant)
                _add_local(next_probs, next_rew_acc, next_s2, self.gamma * (1.0 - self.alpha), float(h))
                _add_local(next_probs, next_rew_acc, next_s3, (1.0 - self.gamma) * (1.0 - self.alpha), 0.0)
            else:
                _add_local(next_probs, next_rew_acc, state, 1.0, float(self.error_penalty) * 0.05)

        elif action == self.Action.Censor:
            # Censor does not change chain height much; give small reward if succeed
            _add_local(next_probs, next_rew_acc, (a, h, fork), 1.0, 0.1)

        elif action == self.Action.SelectiveRelay:
            # create temporary effect: increase gamma locally (we emulate by making next state same but add small cost)
            _add_local(next_probs, next_rew_acc, (a, h, fork), 1.0, -0.01)

        elif action == self.Action.Exit:
            # model stake exit by keeping state but giving small negative (opportunity cost)
            _add_local(next_probs, next_rew_acc, (a, h, fork), 1.0, -0.02)

        else:
            # fallback (shouldn't happen)
            _add_local(next_probs, next_rew_acc, state, 1.0, -0.01)

        # ---- finalize: normalize, jitter, and write into transitions ----
        _finalize_and_write(next_probs, next_rew_acc, state)
        return transitions

    # 你可以保留 get_honest_revenue（或改名）以配合训练监控
    def get_honest_revenue(self) -> float:
        return self.alpha * (1 + self.transaction_chance)
