from enum import Enum
from typing import Tuple

from blockchain_mdps import BlockchainModel, StateTransitions
from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.base_space.space import Space


class EthereumPosModel(BlockchainModel):
    def __init__(self, alpha: float, beta: float, gamma: float, max_fork: int, transaction_chance: float, max_pool: int):
        self.alpha = alpha  # 攻击者的挖矿概率
        self.beta = beta    # 攻击者的作恶概率
        self.gamma = gamma  # 诚实验证者的共识概率
        self.max_fork = max_fork  # 允许的最大分叉链数量
        self.transaction_chance = transaction_chance  # 交易包含的概率
        self.max_pool = max(max_pool, max_fork)  # 交易池的最大容量

        self.block_reward = 1  # 区块奖励

        # 枚举类型，定义不同状态和动作
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Adopt', 'Reveal', 'Mine'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def get_state_space(self) -> Space:
        # 构建状态空间，表示不同链条的状态和其他系统参数
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool),
                                                                           (0, self.max_fork), (0, self.max_fork),
                                                                           (0, self.max_pool), (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        # 动作空间：表示攻击者和诚实者可以采取的行动
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_fork))

    def get_initial_state(self) -> BlockchainModel.State:
        # 初始状态：两条空链（没有区块和交易），没有分叉链，池中无交易
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (0,) * 5

    def get_final_state(self) -> BlockchainModel.State:
        # 最终状态：所有链都清空，且标记为无效状态
        return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (-1,) * 5

    def dissect_state(self, state: BlockchainModel.State) -> Tuple[tuple, tuple, Enum, int, int, int, int, int]:
        # 解剖状态，将状态拆分为具体的元素，用于后续的状态分析和转移
        a = state[:2 * self.max_fork]  # 攻击者链
        h = state[2 * self.max_fork:4 * self.max_fork]  # 诚实者链
        fork = state[-6]  # 分叉状态
        pool = state[-5]  # 交易池状态
        length_a = state[-4]  # 攻击者链长度
        length_h = state[-3]  # 诚实者链长度
        transactions_a = state[-2]  # 攻击者链上的交易数量
        transactions_h = state[-1]  # 诚实者链上的交易数量

        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h

    def construct_state(self, a: tuple, h: tuple, fork: Enum, pool: int,
                        length_a: int, length_h: int, transactions_a: int,
                        transactions_h: int) -> BlockchainModel.State:
        """
        构建新的状态，基于当前攻击者链和诚实者链的状态，以及其他参数。

        :param a: 攻击者链
        :param h: 诚实者链
        :param fork: 分叉状态
        :param pool: 交易池的状态
        :param length_a: 攻击者链长度
        :param length_h: 诚实者链长度
        :param transactions_a: 攻击者链上的交易数量
        :param transactions_h: 诚实者链上的交易数量
        :return: 新的状态
        """
        return *a, *h, fork, pool, length_a, length_h, transactions_a, transactions_h

    def create_empty_chain(self) -> tuple:
        # 创建一个空链（没有区块和交易）
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:
        # 判断链的有效性
        if len(chain) != self.max_fork * 2:
            return False

        valid_parts = sum(isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
                          for block, transaction in zip(chain[::2], chain[1::2]))
        if valid_parts < self.max_fork:
            return False

        # 链应当从区块开始，没有区块的地方不应包含交易
        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min([self.max_fork - 1]
                             + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        invalid_transactions = sum(block is self.Block.NoBlock and transaction is self.Transaction.With
                                   for block, transaction in zip(chain[::2], chain[1::2]))
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        # 验证整个状态的有效性，确保链和其他参数都满足约束条件
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        return self.is_chain_valid(a) and self.is_chain_valid(h) \
            and length_a == self.chain_length(a) \
            and length_h == self.chain_length(h) \
            and transactions_a == self.chain_transactions(a) <= pool \
            and transactions_h == self.chain_transactions(h) <= pool

    def chain_length(self, chain: tuple) -> int:
        # 计算链的长度，即存在的区块数量
        return len([block for block in chain[::2] if block is self.Block.Exists])

    def chain_transactions(self, chain: tuple) -> int:
        # 计算链上交易的数量
        return len([block for block, transaction in zip(chain[::2], chain[1::2])
                    if block is self.Block.Exists and transaction is self.Transaction.With])

    def choose_chain_based_on_weight(self, a: tuple, h: tuple) -> str:
        # 计算每条链的权重（基于主链和分叉链的深度）
        weight_a = self.calculate_chain_weight(a)
        weight_h = self.calculate_chain_weight(h)

        # 比较权重，返回权重较大的链
        return 'a' if weight_a > weight_h else 'h'

    def calculate_chain_weight(self, chain: tuple) -> int:
        # 假设权重由主链深度和分叉链深度共同决定
        main_chain_length = self.chain_length(chain)
        fork_depth = sum([block == self.Block.Exists for block in chain[::2]]) - main_chain_length
        return main_chain_length + fork_depth

    def calculate_reward(self, chain_type: str, transactions_a: int, transactions_h: int) -> int:
        # 基于选择的链类型和其上的交易数量计算奖励
        if chain_type == 'a':
            return transactions_a * self.block_reward
        else:
            return transactions_h * self.block_reward

    def add_block(self, chain: tuple, add_transaction: bool) -> tuple:
        # 在链上添加一个新的区块，并决定是否包含交易
        transaction = self.Transaction.With if add_transaction else self.Transaction.NoTransaction
        index = self.chain_length(chain)
        chain = list(chain)
        chain[2 * index] = self.Block.Exists
        chain[2 * index + 1] = transaction
        return tuple(chain)

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        # 将链向后移动，用于对齐新添加的区块
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.get_final_state(), probability=1, reward=self.error_penalty)
            return transitions

        elif state == self.get_final_state():
            transitions.add(self.get_final_state(), probability=1)
            return transitions

        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        action_type, action_param = action

        # 处理不同的动作类型
        if action_type == self.Action.Illegal:
            # 非法操作会直接导致惩罚，转到最终状态
            transitions.add(self.get_final_state(), probability=1, reward=self.error_penalty)

        elif action_type == self.Action.Adopt:
            # 采纳操作，选择主链，并计算权重和发放奖励
            new_chain = self.choose_chain_based_on_weight(a, h)
            reward = self.calculate_reward(new_chain, transactions_a, transactions_h)

            # 根据新链更新状态
            if new_chain == 'a':
                new_a = self.add_block(a, True)
                transitions.add(self.construct_state(new_a, h, fork, pool, length_a + 1, length_h, transactions_a + 1,
                                                     transactions_h),
                                probability=self.gamma, reward=reward)
            else:
                new_h = self.add_block(h, True)
                transitions.add(self.construct_state(a, new_h, fork, pool, length_a, length_h + 1, transactions_a,
                                                     transactions_h + 1),
                                probability=self.gamma, reward=reward)

        elif action_type == self.Action.Reveal:
            # 揭露操作，攻击者可能揭露隐藏的链
            # （在这里，你可以添加复杂的揭露逻辑，判断何时揭露，如何影响状态）
            # 示例：假设攻击者揭露了一个比诚实链更长的链
            if length_a > length_h:
                new_h = a
                transitions.add(
                    self.construct_state(a, new_h, fork, pool, length_a, length_a, transactions_a, transactions_a),
                    probability=self.alpha, reward=self.block_reward)

        elif action_type == self.Action.Mine:
            # 挖矿操作，创建一个新的区块
            if action_param < self.max_fork:
                new_a = self.add_block(a, False)
                transitions.add(
                    self.construct_state(new_a, h, fork, pool, length_a + 1, length_h, transactions_a, transactions_h),
                    probability=self.beta, reward=self.block_reward)
            else:
                new_h = self.add_block(h, False)
                transitions.add(
                    self.construct_state(a, new_h, fork, pool, length_a, length_h + 1, transactions_a, transactions_h),
                    probability=self.gamma, reward=self.block_reward)

        return transitions
