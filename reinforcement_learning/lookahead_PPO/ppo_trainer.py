from blockchain_mdps import BlockchainModel
from .ppo_algorithm import PPOAlgorithm
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        super().__init__(blockchain_model, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return PPOAlgorithm(**self.creation_args)

