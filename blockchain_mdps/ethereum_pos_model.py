from typing import Tuple
from enum import Enum

from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.state_transitions import StateTransitions
from blockchain_mdps.base.blockchain_model import BlockchainModel


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
        Decompose the state into its constituent components.
        """
        a = state[:2 * self.max_fork]
        h = state[2 * self.max_fork:4 * self.max_fork]
        fork, pool, length_a, length_h, transactions_a, transactions_h = state[-6:]
        return a, h, fork, pool, length_a, length_h, transactions_a, transactions_h

    def create_empty_chain(self) -> tuple:
        return (self.Block.NoBlock, self.Transaction.NoTransaction) * self.max_fork

    def is_chain_valid(self, chain: tuple) -> bool:
        """
        Validate the structural integrity of a chain.
        """
        # Ensure the chain has a valid length
        if len(chain) != self.max_fork * 2:
            return False

        valid_parts = sum(
            isinstance(block, self.Block) and isinstance(transaction, self.Transaction)
            for block, transaction in zip(chain[::2], chain[1::2])
        )
        if valid_parts < self.max_fork:
            return False

        # Check continuity and transaction rules
        last_block = max([0] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.Exists])
        first_no_block = min(
            [self.max_fork - 1] + [idx for idx, block in enumerate(chain[::2]) if block is self.Block.NoBlock])
        if last_block > first_no_block:
            return False

        invalid_transactions = sum(
            block is self.Block.NoBlock and transaction is self.Transaction.With
            for block, transaction in zip(chain[::2], chain[1::2])
        )
        if invalid_transactions > 0:
            return False

        return True

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)
        return self.is_chain_valid(a) and self.is_chain_valid(h) \
            and length_a == self.chain_length(a) \
            and length_h == self.chain_length(h) \
            and transactions_a == self.chain_transactions(a) <= pool \
            and transactions_h == self.chain_transactions(h) <= pool

    @staticmethod
    def truncate_chain(chain: tuple, truncate_to: int) -> tuple:
        return chain[:2 * truncate_to]

    def shift_back(self, chain: tuple, shift_by: int) -> tuple:
        return chain[2 * shift_by:] + (self.Block.NoBlock, self.Transaction.NoTransaction) * shift_by

    def chain_length(self, chain: tuple) -> int:
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

    def flatten_state(self, state) -> tuple:
        flat_state = []
        if isinstance(state, (tuple, list)):
            for element in state:
                flat_state.extend(self.flatten_state(element))
        else:
            flat_state.append(state)
        return tuple(flat_state)

    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1)
            return transitions

        elif state == self.final_state:
            transitions.add(self.final_state, probability=1)
            return transitions

        a, h, fork, pool, length_a, length_h, transactions_a, transactions_h = self.dissect_state(state)

        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)

        if action == self.Action.Withhold:
            if length_a < self.max_fork and length_h < self.max_fork:
                # Add block to attacker's chain
                new_a = self.add_block(a, False)
                attacker_block = (
                        new_a + h + (
                    self.Fork.Irrelevant, pool, self.chain_length(new_a), length_h, transactions_a, transactions_h))
                transitions.add(attacker_block, probability=self.alpha)

                # Add block to honest chain
                new_h = self.add_block(h, False)
                honest_block = (
                        a + new_h + (
                    self.Fork.Relevant, pool, length_a, self.chain_length(new_h), transactions_a, transactions_h))
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1)

        elif action == self.Action.Release:
            if length_a > length_h:
                # Adjust chain lengths using truncate_chain
                new_a = self.shift_back(a, length_h)
                next_state = (new_a + self.create_empty_chain() + (self.Fork.Irrelevant, pool,
                                                                   self.chain_length(new_a), 0, transactions_a,
                                                                   transactions_h))
                transitions.add(next_state, probability=1, reward=length_h + 1)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Equivocate:
            if fork == self.Fork.Relevant and length_a < self.max_fork and length_h < self.max_fork:
                fork_1 = self.add_block(h, False)
                fork_2 = self.add_block(a, False)
                next_state_1 = a + fork_1 + (
                    self.Fork.Relevant, pool, length_a, self.chain_length(fork_1), transactions_a, transactions_h)
                next_state_2 = fork_2 + h + (
                    self.Fork.Relevant, pool, self.chain_length(fork_2), length_h, transactions_a, transactions_h)

                transitions.add(next_state_1, probability=0.5)
                transitions.add(next_state_2, probability=0.5)

            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)

        elif action == self.Action.Vote:
            if length_a < self.max_fork and length_h < self.max_fork:
                # Add block to attacker's chain
                add_transaction = transactions_a < pool
                new_a = self.add_block(a, add_transaction)
                attacker_block = (
                        new_a + h + (self.Fork.Irrelevant, pool - int(add_transaction), self.chain_length(new_a),
                                     length_h, transactions_a + int(add_transaction), transactions_h)
                )
                transitions.add(attacker_block, probability=self.alpha)

                # Add block to honest chain
                add_transaction = transactions_h < pool
                new_h = self.add_block(h, add_transaction)
                honest_block = (
                        a + new_h + (self.Fork.Relevant, pool - int(add_transaction), length_a,
                                     self.chain_length(new_h), transactions_a, transactions_h + int(add_transaction))
                )
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1)

        return transitions

    def get_honest_revenue(self) -> float:
        return self.alpha * (1 + self.transaction_chance)


def main():
    model = EthereumPoSModel(
        alpha=0.3,
        gamma=0.5,
        max_fork=10,
        transaction_chance=0.2,
        max_pool=10
    )

    print("Model created:", model)

    # 测试 get_initial_state 和 dissect_state
    initial_state = model.get_initial_state()
    print("\nInitial state:", initial_state)
    dissected = model.dissect_state(initial_state)
    print("Dissected initial state:", dissected)
    print("Dissected components types:", [type(comp) for comp in dissected])

    # 测试 get_final_state
    final_state = model.get_final_state()
    print("\nFinal state:", final_state)

    # 测试链的相关方法
    empty_chain = model.create_empty_chain()
    print("\nEmpty chain:", empty_chain)
    print("Empty chain is valid:", model.is_chain_valid(empty_chain))

    # 测试链的长度计算和交易统计
    chain_with_block = model.add_block(empty_chain, add_transaction=True)
    print("\nChain with one block added:", chain_with_block)
    print("Chain length:", model.chain_length(chain_with_block))
    print("Chain transactions:", model.chain_transactions(chain_with_block))
    print("Chain is valid:", model.is_chain_valid(chain_with_block))

    # 测试状态的合法性检查
    # valid_state = initial_state[:2 * model.max_fork] + chain_with_block[2 * model.max_fork:]  # 构造有效状态
    valid_state = model.get_initial_state()
    print("\nValid state created:", valid_state)
    print("State is valid:", model.is_state_valid(valid_state))

    # 测试 get_state_transitions
    transitions = model.get_state_transitions(valid_state, model.Action.Release)
    print("\nState transitions (Release action):")

    # 打印 action 和 fork 的枚举值
    print("\nAvailable Actions:", list(model.Action))
    print("Fork States:", list(model.Fork))


if __name__ == "__main__":
    main()
