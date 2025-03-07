import logging
from typing import Tuple
from enum import Enum

from blockchain_mdps.base.base_space.default_value_space import DefaultValueSpace
from blockchain_mdps.base.base_space.discrete_space import DiscreteSpace
from blockchain_mdps.base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from blockchain_mdps.base.state_transitions import StateTransitions
from blockchain_mdps.base.blockchain_model import BlockchainModel


class EthereumPoSModel(BlockchainModel):
    def __init__(self, alpha: float, gamma: float, max_fork: int, fee: float, transaction_chance: float, max_pool: int):
        self.alpha = alpha  # 恶意验证者比例
        self.gamma = gamma  # 区块传播延迟概率
        self.max_fork = max_fork
        self.fee = fee
        self.transaction_chance = transaction_chance
        self.max_pool = max(max_pool, max_fork)

        # self.block_reward = 1 / (1 + self.transaction_chance * self.fee)
        # No need for normalization
        self.block_reward = 1

        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Withhold', 'Release', 'Equivocate', 'Vote'])
        self.Block = self.create_int_enum('Block', ['NoBlock', 'Exists'])
        self.Transaction = self.create_int_enum('Transaction', ['NoTransaction', 'With'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}' \
               f'({self.alpha}, {self.gamma}, {self.max_fork}, {self.fee}, {self.transaction_chance}, {self.max_pool})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.gamma, self.max_fork, self.fee, self.transaction_chance, self.max_pool)

    def get_state_space(self):
        elements = [self.Block, self.Transaction] * (2 * self.max_fork) + [self.Fork, (0, self.max_pool),
                                                                           (0, self.max_fork), (0, self.max_fork),
                                                                           (0, self.max_pool), (0, self.max_pool)]
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self):
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        # return self.create_empty_chain() * 2 + (self.Fork.Irrelevant,) + (0,) * 5
        return self.create_empty_chain() * 2 + (self.Fork.Relevant,) + (0,) * 5

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
        reward = 0

        # if length_h == self.max_fork:
        # print("Current State is: ", state)
        # a = self.create_empty_chain()
        # h = self.create_empty_chain()
        # length_a = length_h = 0
        # new_state = (a + h + (self.Fork.Relevant, pool, length_a, length_h, transactions_a, transactions_h))
        # print("After state is: ", new_state)
        # next_state = (self.create_empty_chain() + self.create_empty_chain() + (self.Fork.Relevant, pool,
        #                                                                        0, 0, transactions_a,
        #                                                                        transactions_h))
        # transitions.add(next_state, probability=1, reward=0)

        # state before final state is 0, 0, 10, 0, 0
        # find ways to reduce honest chain length
        # so that it won't extend the maximum

        if action is self.Action.Illegal:
            reward = self.error_penalty / 2
            transitions.add(self.final_state, probability=1, reward=reward)

            if reward != 0:
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"[Step Reward] Action={action} Reward={reward:.4f}")
                with open("reward_record.txt", "w") as f:
                    f.write(f"[Step Reward] Action={action} Reward={reward:.4f}" + "\n")

        if action == self.Action.Withhold:
            if length_h >= length_a or length_h == self.max_fork or length_a == self.max_fork:
                new_state = (self.create_empty_chain() + self.create_empty_chain() +
                             (self.Fork.Relevant, pool, 0, 0, 0, 0))
                transitions.add(new_state, probability=1, reward=reward)
            elif length_a < self.max_fork and length_h < self.max_fork:
                add_transaction = transactions_a < pool
                # Add block to attacker's chain
                new_a = self.add_block(a, add_transaction)
                # new_a = a
                attacker_block = (
                        new_a + h + (self.Fork.Relevant, pool, self.chain_length(new_a), length_h,
                                     transactions_a + int(add_transaction), transactions_h))
                transitions.add(attacker_block, probability=self.alpha)

                add_transaction = transactions_h < pool
                # Add block to honest chain
                new_h = self.add_block(h, add_transaction)
                # new_h = h
                honest_block = (
                        a + new_h + (self.Fork.Relevant, pool, length_a, self.chain_length(new_h),
                                     transactions_a, transactions_h + int(add_transaction)))
                transitions.add(honest_block, probability=1 - self.alpha)
            else:
                transitions.add(self.final_state, probability=1)

            if reward != 0:
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"[Step Reward] Action={action} Reward={reward:.4f}")
                with open("reward_recode.txt", "w") as f:
                    f.write(f"[Step Reward] Action={action} Reward={reward:.4f}" + "\n")

        elif action == self.Action.Release:
            if length_a >= length_h:
                # Adjust chain lengths using truncate_chain
                new_a = self.shift_back(a, length_h)
                accepted_blocks = length_h
                accepted_transactions = self.chain_transactions(self.truncate_chain(a, accepted_blocks))
                reward = (accepted_blocks + accepted_transactions * self.fee) * self.block_reward
                next_state = (new_a + self.create_empty_chain() + (self.Fork.Irrelevant, pool - accepted_transactions,
                                                                   self.chain_length(new_a), 0,
                                                                   transactions_a - accepted_transactions,
                                                                   transactions_h))
                transitions.add(next_state, probability=1, reward=reward)
            else:
                new_h = self.shift_back(h, length_a)
                accepted_blocks = length_a
                accepted_transactions = self.chain_transactions(self.truncate_chain(h, accepted_blocks))
                reward = 0
                next_state = (self.create_empty_chain() + new_h + (self.Fork.Relevant, pool - accepted_transactions,
                                                                   0, self.chain_length(new_h),
                                                                   transactions_a,
                                                                   transactions_h - accepted_transactions))
                transitions.add(next_state, probability=1, reward=reward)

                if reward != 0:
                    logging.basicConfig(level=logging.INFO)
                    logger = logging.getLogger(__name__)
                    logger.info(f"[Step Reward] Action={action} Reward={reward:.4f}")
                    with open("reward_recode.txt", "w") as f:
                        f.write(f"[Step Reward] Action={action} Reward={reward:.4f}" + "\n")

        elif action == self.Action.Equivocate:
            if (fork == self.Fork.Relevant and length_a < self.max_fork and length_h < self.max_fork
                    and length_a + length_h > 0):
                success_probability = length_a / (length_a + length_h)
                next_state_1 = a + self.create_empty_chain() + (
                    self.Fork.Relevant, pool, length_a, 0,
                    transactions_a, 0)
                next_state_2 = self.create_empty_chain() + h + (
                    self.Fork.Relevant, pool, 0, length_h,
                    0, transactions_h)

                transitions.add(next_state_1, probability=success_probability, reward=reward)
                transitions.add(next_state_2, probability=1 - success_probability, reward=reward)

            else:
                block = self.create_empty_chain() + self.create_empty_chain() + (self.Fork.Relevant, pool, 0, 0, 0, 0)
                transitions.add(block, probability=1, reward=self.error_penalty)

            if reward != 0:
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"[Step Reward] Action={action} Reward={reward:.4f}")
                with open("reward_recode.txt", "w") as f:
                    f.write(f"[Step Reward] Action={action} Reward={reward:.4f}" + "\n")

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
                block = a + h + (self.Fork.Relevant, pool, length_a, length_h, transactions_a, transactions_h)
                transitions.add(block, probability=1)

            if reward != 0:
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"[Step Reward] Action={action} Reward={reward:.4f}")
                with open("reward_recode.txt", "w") as f:
                    f.write(f"[Step Reward] Action={action} Reward={reward:.4f}" + "\n")

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
