from abc import ABC

from blockchain_mdps import BlockchainModel
from .graph_drawing_callback import GraphDrawingCallback
from .mcts_algorithm import MCTSAlgorithm
from .mcts_tensorboard_logging_callback import MCTSTensorboardLoggingCallback
from ..base.experience_acquisition.experience import Experience
from ..base.training.callbacks.composition_callback import CompositionCallback
from ..base.training.callbacks.training_callback import TrainingCallback
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer

import random
import torch.nn as nn
import torch


class DummyApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))  # 只需一个占位参数，避免 .parameters() 为空

    def forward(self, x):
        return x

    def update(self, state_dict):
        """确保 state_dict 是 dict"""
        if isinstance(state_dict, DummyApproximator):
            state_dict = state_dict.state_dict()  # 提取 state_dict

        self.load_state_dict(state_dict)  # 加载权重


class MCTSTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, visualize_every_episode: bool = False,
                 visualize_every_step: bool = False, **kwargs) -> None:
        self.visualize_every_episode = visualize_every_episode
        self.visualize_every_step = visualize_every_step
        super().__init__(blockchain_model, use_bva=True, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        if self.creation_args.get('algorithm_type') == 'greedy':
            return GreedyAlgorithm(blockchain_model=self.blockchain_model, **self.creation_args)
        elif self.creation_args.get('algorithm_type') == 'random':
            return RandomAlgorithm(blockchain_model=self.blockchain_model, **self.creation_args)
        else:
            return MCTSAlgorithm(blockchain_model=self.blockchain_model, **self.creation_args)

    def create_callback(self) -> TrainingCallback:
        callbacks = [super().create_callback(), MCTSTensorboardLoggingCallback()]

        if self.visualize_every_episode or self.visualize_every_step:
            callbacks.append(GraphDrawingCallback(visualize_every_episode=self.visualize_every_episode,
                                                  visualize_every_step=self.visualize_every_step))

        return CompositionCallback(*callbacks)


class GreedyAgent:
    def __init__(self, blockchain_model: BlockchainModel, approximator=None, simulator=None):
        self.blockchain_model = blockchain_model
        self.approximator = approximator or DummyApproximator()
        self.simulator = simulator
        self.base_value_approximation = 0.0
        self.mc_trajectory_lengths = []
        self.monte_carlo_tree_nodes = {}
        self.depth = 1

    def select_action(self, state):
        # 选择当前状态下最优的确定性动作
        possible_actions = self.blockchain_model.get_possible_actions(state)
        best_action = max(possible_actions, key=lambda action: self.blockchain_model.get_reward(state, action))
        return best_action

    def reset(self, keep_state: bool = False):
        # greedy 策略通常不维护状态，所以这里不需要做什么
        pass

    def update(self, *args, **kwargs):
        # 可选调试输出
        # print(f"[GreedyAgent] Ignoring update call with args={args}, kwargs={kwargs}")
        pass

    def step(self, explore=True):
        # 这里我们假设dummy值需要用torch.Tensor来表示
        dummy_action = 0
        dummy_reward = 0
        dummy_next_state = torch.tensor([0.0])  # next_state一般是一个张量

        # 构建info字典，确保包含revenue键
        info = {
            'revenue': 0.0,  # 添加revenue
            'length': 0,  # 假设是回合的长度
            'actions': {}  # 假设actions字典是空的
        }

        return Experience(
            action=dummy_action,
            reward=dummy_reward,
            next_state=dummy_next_state,
            prev_state=torch.tensor([0.0]),  # prev_state需要是torch.Tensor类型
            difficulty_contribution=0,
            prev_difficulty_contribution=0,
            is_done=True,
            legal_actions=torch.tensor([0]),  # legal_actions通常是一个张量
            target_value=torch.tensor([0.0]),  # target_value也是一个张量
            info=info  # info是一个字典
        )


class GreedyAlgorithm(RLAlgorithm, ABC):
    def __init__(self, blockchain_model: BlockchainModel, **creation_args) -> None:
        self.blockchain_model = blockchain_model
        super().__init__(**creation_args)

    def create_agent(self) -> GreedyAgent:
        return GreedyAgent(
            blockchain_model=self.blockchain_model,
            approximator=self.approximator,  # RLAlgorithm 父类里定义了 self.approximator
            simulator=self.blockchain_model  # 如果 simulator 可以复用 blockchain_model
        )

    def create_approximator(self):
        # Greedy 策略不使用近似函数，返回 None
        return DummyApproximator()

    def create_loss_fn(self):
        # Greedy 策略不需要损失函数，返回 None
        return None


class RandomAgent:
    def __init__(self, blockchain_model: BlockchainModel):
        self.blockchain_model = blockchain_model

    def select_action(self, state):
        # 随机选择一个动作
        possible_actions = self.blockchain_model.get_possible_actions(state)
        return random.choice(possible_actions)


class RandomAlgorithm(RLAlgorithm, ABC):
    def __init__(self, blockchain_model: BlockchainModel, **creation_args) -> None:
        self.blockchain_model = blockchain_model
        super().__init__(**creation_args)

    def create_agent(self) -> RandomAgent:
        return RandomAgent(self.blockchain_model)

    def create_approximator(self):
        # Random 策略不使用近似函数，返回 None
        return DummyApproximator()

    def create_loss_fn(self):
        # Random 策略不需要损失函数，返回 None
        return None
