from typing import Tuple
from enum import Enum
from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.state_transitions import StateTransitions
from .base.blockchain_model import BlockchainModel


class EthereumPoSModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int, transaction_chance: float, max_pool: int):
        self.alpha = alpha  # 恶意验证者比例
        self.gamma = gamma  # 区块传播延迟概率
        self.max_fork = max_fork
        self.transaction_chance = transaction_chance
        self.max_pool = max(max_pool, max_fork)

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Withhold', 'Release', 'Equivocate', 'Vote'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_fork}, {self.transaction_chance}, {self.max_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.transaction_chance, self.max_pool)

    def get_state_space(self):
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool),
                                                                           (0, self.max_fork), (0, self.max_fork),
                                                                           (0, self.max_pool), (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (0,) * 5

    def get_final_state(self) -> BlockchainModel.State:
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,) * 5

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int, int, int, int, int]:
        """
        解析状态，将状态分解为各个组成部分。
        :param state: 当前状态
        :return: 攻击者链、诚实节点链、分叉状态、交易池大小、攻击者链长度、诚实链长度、攻击者链交易数量、诚实链交易数量
        """
        max_fork = self.max_fork
        # 提取攻击者链和诚实节点链
        a = state[:2 * max_fork]
        h = state[2 * max_fork:4 * max_fork]

        # 提取分叉状态和其他参数
        fork = state[-6]
        pool = state[-5]
        length_a = state[-4]
        length_h = state[-3]
        transactions_a = state[-2]
        transactions_h = state[-1]

        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h

    def create_empty_chain(self) -> tuple:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:
        """
        检查链的结构是否有效，包括长度和区块类型。
        :param chain: 当前链，表示为 (Block, Transaction) 的元组列表
        :return: 链是否合法（True 或 False）
        """
        # 检查链的长度是否为 2 * max_fork，因为每个区块包含 (Block, Transaction)
        if len(chain) != self.max_fork * 2:
            return False

        # 检查链的区块和交易类型是否匹配
        valid_parts = sum(
            isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
            for block, transaction in zip(chain[::2], chain[1::2])
        )
        if valid_parts < self.max_fork:
            return False

        # 确保链中所有 Block.NoBlock 后面的区块都是 Block.NoBlock
        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min(
            [self.max_fork - 1] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        # 检查没有交易存在于 Block.NoBlock 中
        invalid_transactions = sum(
            block is self.Block.NoBlock and transaction is self.Transaction.With
            for block, transaction in zip(chain[::2], chain[1::2])
        )
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        """
        检查状态是否合法，包括链的结构、长度和交易数量是否符合要求。
        :param state: 当前状态
        :return: 状态是否合法（True 或 False）
        """
        # 解析状态
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)

        # 检查链的有效性
        if not self.is_chain_valid(a) or not self.is_chain_valid(h):
            return False

        # 检查链长度是否与状态中的长度一致
        if length_a != self.chain_length(a) or length_h != self.chain_length(h):
            return False

        # 检查交易数量是否不超过交易池大小
        if transactions_a > pool or transactions_h > pool:
            return False

        return True

    @staticmethod
    def truncate_chain(chain: tuple, truncate_to: int) -> tuple:
        return chain[:2 * truncate_to]

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def chain_length(self, chain: tuple) -> int:
        """
        计算链的长度，即包含 Block.Exists 的区块数量。
        :param chain: 当前链，表示为 (Block, Transaction) 的元组列表
        :return: 链的长度（包含 Block.Exists 的区块数量）
        """
        # 使用列表推导来统计链中 Block.Exists 的数量
        return len([block for block in chain[::2] if block is self.Block.Exists])

    def chain_transactions(self, chain: tuple) -> int:
        return len([block for block, transaction in zip(chain[::2], chain[1::2])
                    if block is self.Block.Exists and transaction is self.Transaction.With])

    def add_block(self, chain: tuple, add_transaction: bool) -> tuple:
        transaction = self.Transaction.With if add_transaction else self.Transaction.NoTransaction
        index = self.chain_length(chain)
        chain = list(chain)
        chain[2 * index] = self.Block.Exists
        chain[2 * index + 1] = transaction
        return tuple(chain)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)

        # 使用 chain_length() 获取链的长度
        length_a = self.chain_length(a)
        length_h = self.chain_length(h)

        if action == self.Action.Withhold:
            # 构造完整的状态 `tuple`
            attacker_block = (a + ((self.Block.Exists, self.Transaction.NoTransaction),), h,
                              self.Fork.Irrelevant, pool, length_a + 1, length_h, transactions_a, transactions_h)
            transitions.add(attacker_block, probability=self.alpha)

            honest_block = (a, h + ((self.Block.Exists, self.Transaction.NoTransaction),),
                            self.Fork.Relevant, pool, length_a, length_h + 1, transactions_a, transactions_h)
            transitions.add(honest_block, probability=1 - self.alpha)

        elif action == self.Action.Release:
            if length_a > length_h:
                # 使用 chain_length() 获取长度并更新状态
                next_state = (a[:length_a - length_h - 1], self.create_empty_chain(),
                              self.Fork.Irrelevant, pool, length_a - (length_h + 1), 0, transactions_a, transactions_h)
                transitions.add(next_state, probability=1, reward=length_h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Equivocate:
            if fork == self.Fork.Relevant:
                next_state = (a, h, self.Fork.Active, pool, length_a, length_h, transactions_a, transactions_h)
                transitions.add(next_state, probability=1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Vote:
            attacker_block = (a + ((self.Block.Exists, self.Transaction.NoTransaction),), h,
                              self.Fork.Irrelevant, pool, length_a + 1, length_h, transactions_a, transactions_h)
            transitions.add(attacker_block, probability=self.alpha)

            honest_block = (a, h + ((self.Block.Exists, self.Transaction.NoTransaction),),
                            self.Fork.Relevant, pool, length_a, length_h + 1, transactions_a, transactions_h)
            transitions.add(honest_block, probability=1 - self.alpha)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha * (1 + self.transaction_chance)
