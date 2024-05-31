from abc import ABC

from .sac_agent import SACAgent
from ..base.function_approximation.mlp_approximator import MLPApproximator
from ..base.training.rl_algorithm import RLAlgorithm


class SACAlgorithm(RLAlgorithm, ABC):

    def create_actor_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('actor_hidden_layers_sizes', [256, 256])
        return MLPApproximator(self.device, self.simulator.state_space_dim, self.simulator.num_of_actions,
                               hidden_layers_sizes)

    def create_critic_approximator(self) -> MLPApproximator:
        hidden_layers_sizes = self.creation_args.get('critic_hidden_layers_sizes', [256, 256])
        return MLPApproximator(self.device, self.simulator.state_space_dim, self.simulator.num_of_actions,
                               hidden_layers_sizes)

    def create_agent(self) -> SACAgent:
        # Retrieve parameters from creation_args or set defaults
        hidden_size = self.creation_args.get('hidden_size', 256)
        state_dim = self.simulator.state_space_dim
        action_dim = self.simulator.num_of_actions
        action_space = self.simulator.action_space

        # Create the MLP Approximator instances for the actor and critic networks
        actor_approximator = self.create_actor_approximator()
        critic_approximator = self.create_critic_approximator()

        # Create the SACAgent instance with the correct parameters
        agent = SACAgent(actor_approximator=actor_approximator,
                         critic_approximator=critic_approximator,
                         simulator=self.simulator,
                         state_dim=state_dim,
                         action_dim=action_dim,
                         hidden_size=hidden_size,
                         action_space=action_space)

        return agent

    # In SAC, the loss function is typically part of the update rule and is not separate like in DQN.
    # Therefore, we do not need to create a separate loss function here.
    # The SACAgent will handle the loss calculations internally during the update.
