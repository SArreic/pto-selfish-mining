from blockchain_mdps import BlockchainModel
from .sac_algorithm import SACAlgorithm
from ..base.training.rl_algorithm import RLAlgorithm
from ..base.training.trainer import Trainer


class SACTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        # Assuming that SAC does not use BVA (Bounded Value Approximation) and does not need to plot a heatmap
        super().__init__(blockchain_model, **kwargs)

    def create_algorithm(self) -> RLAlgorithm:
        return SACAlgorithm(**self.creation_args)

    # You might want to override the train method if the training procedure for SAC
    # is significantly different from what is implemented in the Trainer base class.
    # For example, SAC might require different handling of episodes,
    # or additional steps like updating temperature parameters.
    # However, if the differences are minor, you can rely on the inherited train method.
