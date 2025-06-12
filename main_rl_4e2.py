# from torch.distributions import Categorical
# from collections import defaultdict
# from operator import itemgetter
# from typing import Tuple, Optional, Dict
#
# import numpy as np
# import torch
# import torch.nn.functional as F
#
# from blockchain_mdps import BlockchainModel
# from .mct_node import MCTNode
# from ..base.blockchain_simulator.mdp_blockchain_simulator import MDPBlockchainSimulator
# from ..base.experience_acquisition.agents.bva_agent import BVAAgent
# from ..base.experience_acquisition.experience import Experience
# from ..base.experience_acquisition.exploaration_mechanisms.epsilon_greedy_exploration import EpsilonGreedyExploration
# from ..base.experience_acquisition.exploaration_mechanisms.state_dependant_boltzmann_exploration import \
#     StateDependantBoltzmannExploration
# from ..base.experience_acquisition.replay_buffers.ppo_buffer import PPOBuffer
# from ..base.function_approximation.approximator import Approximator
# from ..base.function_approximation.ppo_approximator import PPOApproximator
#
#
# class MCTSAgent(BVAAgent):
#     def __init__(self, approximator: Approximator, simulator: MDPBlockchainSimulator, starting_epsilon: float,
#                  epsilon_step: float, use_boltzmann: bool, boltzmann_temperature: float, target_pi_temperature: float,
#                  puct_const: float, use_action_prior: bool, depth: int, mc_simulations: int, warm_up_simulations: int,
#                  use_base_approximation: bool, ground_initial_state: bool, use_cached_values: bool, value_clip: float,
#                  nn_factor: float, prune_tree_rate: int, root_dirichlet_noise: float,
#                  planning_strategy: str = "ppo"):
#         super().__init__(approximator, simulator, use_cache=False)
#
#         if use_boltzmann:
#             self.exploration_mechanism = StateDependantBoltzmannExploration(boltzmann_temperature)
#         else:
#             self.exploration_mechanism = EpsilonGreedyExploration(starting_epsilon, epsilon_step)
#
#         self.target_pi_temperature = target_pi_temperature
#         self.puct_const = puct_const
#         self.use_action_prior = use_action_prior
#
#         self.depth = depth
#         assert self.depth > 0
#         self.mc_simulations = mc_simulations
#         assert self.mc_simulations > 0
#         self.warm_up_simulations = warm_up_simulations
#
#         self.use_base_approximation = use_base_approximation
#         self.ground_initial_state = ground_initial_state
#         self.use_cached_values = use_cached_values
#         self.value_clip = value_clip
#         assert self.value_clip >= 0
#         self.nn_factor = nn_factor
#         assert self.nn_factor >= 0
#         self.prune_tree_rate = prune_tree_rate
#         assert self.prune_tree_rate >= 0
#         self.root_dirichlet_noise = root_dirichlet_noise
#         assert self.root_dirichlet_noise >= 0
#
#         self.planning_strategy = "ppo"
#         print(self.planning_strategy)
#         assert self.planning_strategy in ["mcts", "greedy", "random", "ppo"]
#
#         if self.planning_strategy == "ppo":
#             state_dim = simulator.state_space_dim
#             action_dim = simulator.num_of_actions
#             self.ppo_approximator = PPOApproximator(state_dim, action_dim)
#             self.policy_net = self.ppo_approximator.policy_net
#             self.value_net = self.ppo_approximator.value_net
#             self.buffer = PPOBuffer(buffer_size=2048)
#             self.optimizer = torch.optim.Adam(
#                 list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=3e-4
#             )
#             self.ppo_epochs = 4
#             self.clip_epsilon = 0.2
#             self.value_coef = 0.5
#             self.entropy_coef = 0.01
#
#         self.mc_trajectory_lengths = []
#
#         self.monte_carlo_tree_nodes: Dict[BlockchainModel.State, MCTNode] = {}
#
#     def __repr__(self) -> str:
#         d = {'type': self.__class__.__name__, 'exploration_mechanism': self.exploration_mechanism, 'depth': self.depth,
#              'simulations': self.mc_simulations, 'ground_initial_state': self.ground_initial_state,
#              'value_clip': self.value_clip, 'nn_factor': self.nn_factor}
#         return str(d)
#
#     def warm_up(self) -> None:
#         for _ in range(self.warm_up_simulations):
#             self.simulate_trajectory(self.simulator.torch_to_tuple(self.current_state), False)
#
#     def plan_action(self, explore: bool = True) -> Tuple[int, torch.Tensor]:
#         state_tuple = self.simulator.torch_to_tuple(self.current_state)
#         if self.planning_strategy == "ppo":
#             with torch.no_grad():
#                 logits = self.policy_net(self.current_state)
#                 legal_actions = self.simulator.get_state_legal_actions_tensor(self.current_state)
#                 if not legal_actions.any():
#                     chosen_action = np.random.randint(self.simulator.num_of_actions)
#                     log_prob = torch.tensor(0.0, device=self.simulator.device)
#                     value = self.value_net(self.current_state)
#                     self.buffer.store(self.current_state, chosen_action, log_prob, value)
#                     target_pi = torch.zeros(self.simulator.num_of_actions, device=self.simulator.device)
#                     target_pi[chosen_action] = 1.0
#                     return chosen_action, torch.cat([value.view(1), target_pi])
#
#                 logits[~legal_actions] = float('-inf')
#                 dist = Categorical(logits=logits)
#                 action = dist.sample()
#                 log_prob = dist.log_prob(action)
#                 value = self.value_net(self.current_state)
#             self.buffer.store(self.current_state, action, log_prob, value)
#             target_pi = torch.zeros(self.simulator.num_of_actions, device=self.simulator.device)
#             target_pi[action] = 1.0
#             return action.item(), torch.cat([value.view(1), target_pi])
#
#         else:
#             # mc_sim_revenues = []
#             for _ in range(self.mc_simulations):
#                 self.simulate_trajectory(state_tuple, explore)
#                 # mc_sim_revenues.append(rev)
#
#             root = self.get_node(state_tuple, explore)
#
#             chosen_action = self.invoke_exploration_mechanism(self.exploration_mechanism, root.mc_estimated_q_values,
#                                                               explore)
#             target_value = torch.tensor(root.mc_estimated_q_values[chosen_action], device=self.simulator.device,
#                                         dtype=torch.float)
#             # target_value = torch.tensor(sum(mc_sim_revenues) / self.mc_simulations, device=self.simulator.device,
#             #                             dtype=torch.float)
#
#             if self.use_base_approximation:
#                 target_value -= self.base_value_approximation
#
#             if self.value_clip > 0:
#                 target_value = torch.clamp(target_value, -self.value_clip, self.value_clip)
#
#             target_value /= self.nn_factor
#
#             action_counts = root.action_counts
#             target_pi = torch.zeros((self.simulator.num_of_actions,), device=self.simulator.device, dtype=torch.float)
#             for action, count in action_counts.items():
#                 target_pi[action] = count
#
#             target_pi = target_pi.pow(1 / self.target_pi_temperature)
#             target_pi /= target_pi.sum()
#
#             return chosen_action, torch.cat([target_value.view(1), target_pi])
#
#     def simulate_trajectory(self, state: BlockchainModel.State, exploring: bool) -> None:
#         trajectory = []
#
#         for depth in range(self.depth):
#             current_node = self.get_node(state, exploring)
#             self.expand_node(state, exploring, depth)
#             action_puct_values = self.calculate_action_puct_values(current_node, depth)
#             action = max(action_puct_values.items(), key=itemgetter(1))[0]
#
#             trajectory.append((state, action))
#
#             current_node.visit_count += 1
#             current_node.action_counts[action] += 1
#
#             state = self.simulator.make_random_transition(current_node.action_state_transitions[action])
#
#             if self.ground_initial_state and self.simulator.is_initial_state(state):
#                 break
#
#         for state, action in reversed(trajectory):
#             self.update_node_action_value(state, action, exploring)
#
#         self.mc_trajectory_lengths.append(len(trajectory))
#
#     def reset(self, state: Optional[BlockchainModel.State] = None, keep_state: bool = False) -> None:
#         self.mc_trajectory_lengths = []
#         super().reset(state, keep_state)
#
#     def get_node(self, state: BlockchainModel.State, exploring: bool) -> MCTNode:
#         if state not in self.monte_carlo_tree_nodes:
#             legal_actions = self.simulator.get_state_legal_actions(state)
#
#             action_state_transitions = {}
#             for action in legal_actions:
#                 action_state_transitions[action] = self.simulator.get_state_transition_values(state, action)
#
#             evaluation = self.get_state_evaluation(state, exploring)
#             approximated_value = np.float32(evaluation[0].item())
#
#             prior_action_probabilities = defaultdict(lambda: 1.0)
#
#             if self.use_action_prior:
#                 for legal_action in legal_actions:
#                     prior_action_probabilities[legal_action] = np.float32(evaluation[legal_action + 1].item())
#
#             self.monte_carlo_tree_nodes[state] = MCTNode(state, legal_actions, action_state_transitions,
#                                                          approximated_value, prior_action_probabilities)
#
#         return self.monte_carlo_tree_nodes[state]
#
#     def expand_node(self, state: BlockchainModel.State, exploring: bool, depth: int) -> None:
#         node = self.get_node(state, exploring)
#         if node.expanded:
#             return
#
#         use_nn = depth == self.depth and not self.use_cached_values
#         for action in node.legal_actions:
#             node.mc_estimated_q_values[action] = self.calculate_mean_action_value(state, action, exploring, use_nn)
#
#         node.expanded = True
#
#     def calculate_mean_action_value(self, state: BlockchainModel.State, action: int, exploring: bool,
#                                     use_nn: bool = False) -> float:
#         transition_values = self.get_node(state, exploring).action_state_transitions[action]
#         total_value = 0
#         for next_state in transition_values.probabilities.keys():
#             value = self.get_node(next_state, exploring).get_value(use_nn)
#
#             if self.ground_initial_state and self.simulator.is_initial_state(next_state):
#                 value = self.get_node(next_state, exploring).approximated_value
#
#             # Discount by difficulty contribution
#             value *= self.calculate_difficulty_contribution_discount(
#                 transition_values.difficulty_contributions[next_state])
#
#             # Add transition reward
#             value += transition_values.rewards[next_state] / self.simulator.expected_horizon
#
#             # Multiply by transition probability
#             value *= transition_values.probabilities[next_state]
#
#             total_value += value
#
#         return total_value
#
#     def calculate_difficulty_contribution_discount(self, difficulty_contribution: float) -> float:
#         return (1 - 1 / self.simulator.expected_horizon) ** difficulty_contribution
#
#     def update_node_action_value(self, state: BlockchainModel.State, action: int, exploring: bool) -> None:
#         node = self.get_node(state, exploring)
#
#         sampled_action_value = self.calculate_mean_action_value(node.state, action, exploring)
#
#         if self.use_cached_values:
#             node.mc_estimated_q_values[action] = sampled_action_value
#         else:
#             action_count = node.action_counts[action]
#             # print(action_count)
#             node.mc_estimated_q_values[action] *= (action_count - 1) / action_count
#             node.mc_estimated_q_values[action] += sampled_action_value / action_count
#
#     def calculate_action_puct_values(self, node: MCTNode, depth: int) -> Dict[int, float]:
#         puct_values = {}
#         num_of_actions = len(node.mc_estimated_q_values)
#         for action, q_value in node.mc_estimated_q_values.items():
#             exploration_factor = self.puct_const
#             if self.use_action_prior:
#                 action_prior_probability = node.prior_action_probabilities[action]
#                 if depth == 0:
#                     # Add Dirichlet noise
#                     action_prior_probability *= 1 - self.root_dirichlet_noise
#                     action_prior_probability += self.root_dirichlet_noise / self.simulator.num_of_actions
#                 exploration_factor *= action_prior_probability
#
#             exploration_factor *= np.sqrt(node.visit_count) / (1 + node.action_counts[action])
#             puct_values[action] = q_value + exploration_factor
#
#         return puct_values
#
#     def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
#         legal_actions_tensor = self.simulator.get_state_legal_actions_tensor(state)
#
#         with torch.no_grad():
#             all_values = self.approximator(state).squeeze()
#             q_values = all_values[:self.simulator.num_of_actions]
#             legal_q_values = q_values.masked_fill_(mask=~legal_actions_tensor, value=float('-inf'))
#             value = legal_q_values.max()
#
#             if self.ground_initial_state and exploring and self.simulator.is_initial_state(state):
#                 value *= 0
#
#             value *= self.nn_factor
#
#             if self.value_clip > 0:
#                 value = torch.clamp(value, -self.value_clip, self.value_clip)
#
#             if self.use_base_approximation:
#                 value += self.base_value_approximation
#
#             if not self.use_action_prior:
#                 return value.view(1)
#
#             action_scores = all_values[self.simulator.num_of_actions:]
#             legal_action_scores = action_scores.masked_fill_(mask=~legal_actions_tensor, value=float('-inf'))
#             p_values = torch.nn.functional.softmax(legal_action_scores, dim=0)
#
#             state_evaluation = torch.cat([value.view(1), p_values])
#
#             return state_evaluation
#
#     def step(self, explore: bool = True) -> Experience:
#         if self.planning_strategy == "ppo":
#             action, _ = self.plan_action(explore)
#             experience = self.simulator.step(action)
#             self.buffer.store_reward(experience.reward, experience.is_done)
#             self.current_state = experience.next_state
#             return experience
#
#         else:
#             exp = super().step(explore)
#
#             if self.prune_tree_rate > 0 and self.step_idx % self.prune_tree_rate == 0:
#                 self.prune_tree()
#
#             return exp
#
#     def prune_tree(self) -> None:
#         self.monte_carlo_tree_nodes = {state: node for state, node in self.monte_carlo_tree_nodes.items() if
#                                        node.visit_count > 0}
#
#     def update(self, approximator: Optional[Approximator] = None, base_value_approximation: Optional[float] = None,
#                **kwargs) -> None:
#         if self.planning_strategy == "ppo":
#             if not self.buffer.ready():
#                 return
#             self.buffer.compute_advantages()
#             for _ in range(self.ppo_epochs):
#                 for states, actions, old_log_probs, returns, advantages in self.buffer.iterate_batches():
#                     logits = self.policy_net(states)
#                     values = self.value_net(states).squeeze(-1)
#                     dist = Categorical(logits=logits)
#                     entropy = dist.entropy().mean()
#                     new_log_probs = dist.log_prob(actions)
#
#                     ratio = torch.exp(new_log_probs - old_log_probs)
#                     clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
#                     policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
#                     value_loss = F.mse_loss(values, returns)
#                     loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
#
#                     self.optimizer.zero_grad()
#                     loss.backward()
#                     self.optimizer.step()
#             self.buffer.reset()
#         else:
#             super().update(approximator, base_value_approximation, **kwargs)
#             self.monte_carlo_tree_nodes = {}
#             self.warm_up()
#
#     def reduce_to_v_table(self) -> torch.Tensor:
#         # Set the number of simulations to speed things up
#         original_mc_simulations = self.mc_simulations
#         self.mc_simulations = 1
#
#         # Compute the V table
#         v_table = super().reduce_to_v_table()
#
#         # Set the number back
#         self.mc_simulations = original_mc_simulations
#
#         return v_table
