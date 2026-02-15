from blockchain_mdps import BlockchainModel
from .hp3o_algorithm import HP3OAlgorithm
from .hp3o_orchestrator import HP3OOrchestrator
from ..base.training.trainer import Trainer

class HP3OTrainer(Trainer):
    def __init__(self, blockchain_model: BlockchainModel, **kwargs) -> None:
        super().__init__(blockchain_model, use_bva=True, **kwargs)

    def create_algorithm(self) -> HP3OAlgorithm:
        return HP3OAlgorithm(**self.creation_args)

    def create_orchestrator(self) -> HP3OOrchestrator:
        # We use a custom orchestrator for HP3O to handle trajectory-based updates
        return HP3OOrchestrator(
            algorithm=self.algorithm,
            loggers=self.loggers,
            callback=self.callback,
            blockchain_model=self.blockchain_model,
            expected_horizon=self.expected_horizon,
            random_seed=self.random_seed,
            **self.creation_args
        )
