import argparse
import itertools
import signal
import sys
from pathlib import Path
from typing import Tuple, Any, List, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import single

from blockchain_mdps import *
from blockchain_mdps.avalanche_fee_model import EthereumPosModel
from blockchain_mdps.avalanche_model import AvalancheAttackModel
from blockchain_mdps.base.blockchain_mdps.blockchain_mdp import BlockchainMDP
from blockchain_mdps.ethereum2_fee_model import Ethereum2FeeModel
from blockchain_mdps.ethereum2_model import Ethereum2Model
from reinforcement_learning import *
# noinspection PyUnusedLocal
from reinforcement_learning.base.training.callbacks.bva_callback import BVACallback


# from rl_log_to_graph import create_performance_figure, create_q_values_graph, create_speed_graph


# noinspection PyUnusedLocal
def interrupt_handler(signum: int, frame: Any) -> None:
    # For exiting gracefully
    print(f'Process interrupted with signal {signum}')
    raise InterruptedError('Process interrupted')


def solve_mdp_exactly(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    expected_horizon = int(1e4)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    p, r, _, _ = solver.calc_opt_policy(epsilon=1e-7, max_iter=int(1e10))  # Singular Matrix Detected Here
    sys.stdout.flush()
    revenue = solver.mdp.calc_policy_revenue(p)
    return np.float32(revenue), p


def solve_mdp_randomly(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    np.random.seed(42)  # For reproducibility
    expected_horizon = int(1)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    policy = tuple(np.random.choice(solver.mdp.num_of_actions) for _ in range(solver.mdp.num_of_states))
    revenue = solver.mdp.calc_policy_revenue(policy)
    return np.float32(revenue), policy


def solve_mdp_greedily(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    expected_horizon = int(1e1)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    policy = []
    for state in range(solver.mdp.num_of_states):
        best_action = None
        best_reward = -np.inf
        for action in range(solver.mdp.num_of_actions):
            reward = solver.mdp.R.get_val(action, state, state)
            if reward > best_reward:
                best_reward = reward
                best_action = action
        policy.append(best_action)
    policy = Tuple[np.array(policy)]
    revenue = solver.mdp.calc_policy_revenue(policy)
    return np.float32(revenue), tuple(policy)


def solve_mdp_brute_force(mdp: BlockchainModel) -> Tuple[float, BlockchainModel.Policy]:
    expected_horizon = int(1)
    solver = PTOSolver(mdp, expected_horizon=expected_horizon)
    best_policy = None
    best_revenue = -np.inf
    for policy in itertools.product(range(solver.mdp.num_of_actions), repeat=solver.mdp.num_of_states):
        revenue = solver.mdp.calc_policy_revenue(policy)
        if revenue > best_revenue:
            best_revenue = revenue
            best_policy = policy
    return np.float32(best_revenue), best_policy


def solve_mdp_approx(mdp: BlockchainModel, method: str = 'greedy') -> Tuple[float, BlockchainModel.Policy]:
    if method == 'random':
        return solve_mdp_randomly(mdp)
    elif method == 'greedy':
        return solve_mdp_greedily(mdp)
    elif method == 'brute':
        return solve_mdp_brute_force(mdp)
    else:
        raise ValueError("Unknown method: choose from 'random', 'greedy', 'brute_force'.")


def log_solution_info(mdp: BlockchainModel, rev: float, trainer: Trainer) -> float:
    policy_rev = PTOSolver(mdp).mdp.calc_policy_revenue(trainer.orchestrator.agent.reduce_to_policy())
    trainer.log_info(f'Number of states: {mdp.state_space.size}')
    trainer.log_info(f'Optimal Revenue in ARR-MDP: {rev}')
    trainer.log_info(f'Policy Revenue in ARR-MDP: {policy_rev}')

    return policy_rev


def solve_pt(args: argparse.Namespace) -> None:
    # mdp = EthereumModel(alpha=0.35, max_fork=200)
    mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=20)
    # mdp = SimpleModel(alpha=0.35, max_fork=200)
    # mdp = BitcoinFeeModel(alpha=0.35, gamma=0.5, max_fork=2, fee=2, transaction_chance=0.1, max_pool=2)
    smart_init = 0.35
    print(f'{mdp.state_space.size:,}')
    rev, _ = solve_mdp_exactly(mdp)

    trainer = LDDQNTrainer(mdp, orchestrator_type='synced_multi_process', build_info=args.build_info, random_seed=0,
                           output_root=args.output_root, expected_horizon=10_000, depth=10, batch_size=100,
                           dropout=0, starting_epsilon=0.01, epsilon_step=0, bva_smart_init=smart_init,
                           num_of_episodes_for_average=100, learning_rate=2e-4, nn_factor=0.001, huber_beta=1,
                           epoch_length=10_000, num_of_epochs=1000, replay_buffer_size=20_000,
                           use_base_approximation=True, ground_initial_state=False,
                           train_episode_length=1000, evaluate_episode_length=1000,
                           number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                           # number_of_evaluation_agents=1, number_of_training_agents=1,
                           lower_priority=args.no_bg, bind_all=args.bind_all, stop_goal=rev,
                           output_value_heatmap=False, normalize_target_values=True, use_cached_values=False)

    trainer.run()

    log_solution_info(mdp, rev, trainer)


def solve_pt_multiple_times(args: argparse.Namespace, times: int = 10) -> None:
    # mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=200)
    mdp = SimpleModel(alpha=0.35, max_fork=200)
    print(f'{mdp.state_space.size:,}')
    rev, _ = solve_mdp_exactly(mdp)

    solution_revenues = np.zeros(times)
    simulation_values = pd.DataFrame(columns=['Iteration', 'Run', 'Base Value'])

    for run_index in range(times):
        print(f'Training Iteration {run_index}')
        trainer = LDDQNTrainer(mdp, orchestrator_type='multi_process', build_info=args.build_info,
                               output_root=args.output_root, expected_horizon=1e4, depth=10, batch_size=100,
                               dropout=0.0, starting_epsilon=0.01, epsilon_step=0,
                               num_of_episodes_for_average=20, learning_rate=2e-4, nn_factor=0.001, huber_beta=1,
                               epoch_length=10000, epoch_size=10000, num_of_epochs=150, replay_buffer_size=20000,
                               use_base_approximation=True, ground_initial_state=False,
                               train_episode_length=10000, evaluate_episode_length=10000,
                               number_of_evaluation_agents=args.eval_agents,
                               number_of_training_agents=args.train_agents,
                               # number_of_evaluation_agents=1, number_of_training_agents=1,
                               lower_priority=args.no_bg, bind_all=args.bind_all, stop_goal=rev,
                               normalize_target_values=True)

        trainer.run()

        bva_callback = trainer.get_callbacks_of_type(BVACallback)[0]

        num_of_epochs = len(bva_callback.epoch_history)
        df = pd.DataFrame({'Iteration': range(num_of_epochs),
                           'Run': [run_index] * num_of_epochs,
                           'Base Value': bva_callback.epoch_history})

        simulation_values = simulation_values.append(df, ignore_index=True)

        # for index, value in enumerate(bva_callback.epoch_history):
        #    simulation_values = simulation_values.append({'Iteration': index, 'Run': run_index, 'Base Value': value},
        #                                                 ignore_index=True)

        policy_rev = log_solution_info(mdp, rev, trainer)
        solution_revenues[run_index] = policy_rev

    fig, ax = plt.subplots()
    sns.lineplot(x='Iteration', y='Base Value', data=simulation_values, ax=ax)
    fig.show()
    out_dir = f'{args.output_root}/{args.build_info}/' if args.output_root is not None else ''
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{out_dir}out/run_simulation_values.png')

    fig, ax = plt.subplots()
    sns.histplot(solution_revenues, kde=True, line_kws={'label': 'Agent Policy'}, ax=ax)
    ax.set_xlabel('Final Policy Revenue')
    ax.set_ylabel('Number of Runs')
    limits = ax.get_ylim()
    ax.vlines(x=rev, ymin=limits[0], ymax=limits[1], label='Optimal', linestyles='dashed', colors='k')
    ax.legend()
    # ax.bar(range(len(solution_revenues)), solution_revenues)
    # limits = ax.get_xlim()
    # ax.hlines(y=rev, xmin=limits[0], xmax=limits[1], label='Optimal', linestyles='dashed', colors='k')
    fig.show()
    fig.savefig(f'{out_dir}out/run_solution_values.png')


def solve_sm(args: argparse.Namespace) -> None:
    mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=10)
    print(f'{mdp.state_space.size:,}')
    rev, _ = solve_mdp_exactly(mdp)

    trainer = LSMDQNTrainer(mdp, orchestrator_type='multi_process', build_info=args.build_info,
                            output_root=args.output_root, expected_horizon=1e4, depth=10, batch_size=100,
                            starting_epsilon=0.05, epsilon_step=0, num_of_episodes_for_average=50, learning_rate=2e-5,
                            nn_factor=0.01, huber_beta=1, num_of_epochs=1000,
                            train_episode_length=10000, evaluate_episode_length=1000,
                            use_base_approximation=True, ground_initial_state=False,
                            number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                            lower_priority=args.no_bg, bind_all=args.bind_all, stop_goal=rev)

    trainer.run()

    log_solution_info(mdp, rev, trainer)


def solve_fees(args: argparse.Namespace) -> None:
    mdp = BitcoinFeeModel(alpha=0.35, gamma=0.5, max_fork=10, fee=2, transaction_chance=0.1, max_pool=10)
    print(f'{mdp.state_space.size:,}')

    trainer = LDDQNTrainer(mdp, orchestrator_type='multi_process', build_info=args.build_info,
                           output_root=args.output_root, expected_horizon=1e4, depth=10, batch_size=100,
                           starting_epsilon=0.05, epsilon_step=0, num_of_episodes_for_average=50, learning_rate=2e-5,
                           nn_factor=0.01, huber_beta=1, num_of_epochs=1000,
                           rain_episode_length=10000, evaluate_episode_length=1000,
                           use_base_approximation=True, ground_initial_state=False,
                           number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                           lower_priority=args.no_bg, bind_all=args.bind_all)

    trainer.run()


def test_mcts(args: argparse.Namespace) -> None:
    mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=3)
    print(f'{mdp.state_space.size:,}')
    rev, p = solve_mdp_exactly(mdp)

    trainer = LDDQNTrainer(mdp, orchestrator_type='multi_process', build_info=args.build_info,
                           output_root=args.output_root, expected_horizon=1e4, depth=10, batch_size=100,
                           starting_epsilon=0.05, epsilon_step=0, num_of_episodes_for_average=50, learning_rate=2e-5,
                           nn_factor=0.01, huber_beta=1, num_of_epochs=1000,
                           train_episode_length=10000, evaluate_episode_length=1000,
                           use_base_approximation=True, ground_initial_state=False,
                           number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                           lower_priority=args.no_bg, bind_all=args.bind_all, stop_goal=rev)

    trainer.run()

    solver = PTOSolver(mdp).mdp
    policy_rev = solver.calc_policy_revenue(trainer.orchestrator.agent.reduce_to_policy())

    q_table = trainer.orchestrator.agent.reduce_to_q_table()
    q_table = q_table.detach().clone()

    q_table_policy = trainer.orchestrator.agent.reduce_to_policy_from_q_table(q_table)
    q_table_policy_rev = solver.calc_policy_revenue(q_table_policy)

    trainer.log_info(f'Policy Revenue in ARR-MDP: {policy_rev}')
    trainer.log_info(f'Q Table Policy Revenue in ARR-MDP: {q_table_policy_rev}')

    mdp.print_policy(p, solver.find_reachable_states(p), x_axis=1, y_axis=0, z_axis=2)

    trainer = MCTSTrainer(mdp, build_info=args.build_info, output_root=args.output_root,
                          expected_horizon=1e4, depth=50, batch_size=100, starting_epsilon=0.05, epsilon_step=0,
                          num_of_episodes_for_average=50, learning_rate=2e-4, nn_factor=0.001, huber_beta=1,
                          momentum=0.9, mc_simulations=5, use_base_approximation=True, ground_initial_state=False,
                          num_of_epochs=1000, train_episode_length=10000, evaluate_episode_length=1000,
                          number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                          lower_priority=args.no_bg, bind_all=args.bind_all,
                          cross_entropy_loss_weight=1, puct_const=10, target_pi_temperature=2, stop_goal=rev,
                          use_action_prior=True, use_table=False, visualize_every_episode=False)

    with trainer:
        trainer.run()

    log_solution_info(mdp, rev, trainer)


def run_mcts(args: argparse.Namespace):
    mdp = BitcoinModel(alpha=0.35, gamma=0.5, max_fork=200)
    # mdp = EthereumModel(alpha=0.35, max_fork=20)
    print(f'{mdp.state_space.size:,}')
    # rev, _ = solve_mdp_exactly(mdp)

    trainer = MCTSTrainer(mdp, orchestrator_type='multi_process', build_info=args.build_info,
                          output_root=args.output_root, output_profile=True, expected_horizon=1e4, depth=10,
                          batch_size=100, dropout=0, starting_epsilon=0.01, epsilon_step=0,
                          num_of_episodes_for_average=100, learning_rate=2e-4, nn_factor=0.001, huber_beta=1,
                          epoch_length=10000, epoch_size=10000, num_of_epochs=1000, replay_buffer_size=20000,
                          use_base_approximation=True, ground_initial_state=False, prune_tree_rate=200,
                          train_episode_length=10000, evaluate_episode_length=2000,
                          number_of_evaluation_agents=args.eval_agents, number_of_training_agents=args.train_agents,
                          lower_priority=args.no_bg, bind_all=args.bind_all, output_value_heatmap=False,
                          normalize_target_values=True, use_cached_values=False)

    trainer.run()

    # log_solution_info(mdp, rev, trainer)


def run_mcts_fees(args: argparse.Namespace):
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    max_fork = args.max_fork
    fee = args.fee
    transaction_chance = args.delta
    # simple_mdp = BitcoinModel(alpha=alpha, gamma=gamma, max_fork=max_fork)
    # simple_mdp = Ethereum2Model(alpha=alpha, gamma=gamma, max_stake_pool=max_fork)
    simple_mdp = AvalancheAttackModel(alpha=alpha, beta=beta, max_forks=max_fork, max_depth=max_fork * 2)
    # rev, _ = solve_mdp_approx(simple_mdp, method='random')
    rev, _ = solve_mdp_exactly(simple_mdp)
    print("rev is ", rev)
    print("The best policy is {}".format(_))
    rev = rev if -1 <= rev <= 1 else (-1 if rev < -1 else 1)
    # mdp = Ethereum2FeeModel(alpha=alpha, gamma=gamma, max_pool=max_fork,
    #                         max_fee_pool=max_fork, max_proposals=max_fork,
    #                         max_stake_pool=max_fork, max_votes=max_fork,
    #                         network_congestion=0)
    mdp = EthereumPosModel(alpha=alpha, beta=gamma, gamma=gamma, max_fork=max_fork, max_pool=max_fork,
                           transaction_chance=gamma)

    smart_init = rev * (1 + fee * transaction_chance)
    print("rev is {}, fee is {}, transaction chance is {}".format(rev, fee, transaction_chance))
    print("smart_init is {}".format(smart_init))
    # smart_init = None

    print(f'{mdp.state_space.size:,}')

    trainer = MCTSTrainer(
        mdp,
        orchestrator_type='synced_multi_process',
        build_info=args.build_info,
        output_root=args.output_root,
        output_profile=False,
        output_memory_snapshots=False,
        random_seed=args.seed,
        expected_horizon=10_000,
        depth=5,
        batch_size=100,
        dropout=0,
        length_factor=10,
        starting_epsilon=0.05,
        epsilon_step=0,
        bva_smart_init=smart_init,
        prune_tree_rate=250,
        num_of_episodes_for_average=1000,
        learning_rate=args.lr,
        nn_factor=0.0001,
        mc_simulations=25,
        num_of_epochs=5001,
        epoch_shuffles=2,
        save_rate=100,
        use_base_approximation=True,
        ground_initial_state=False,
        train_episode_length=100,
        evaluate_episode_length=100,
        lr_decay_epoch=1000,
        number_of_training_agents=args.train_agents,
        number_of_evaluation_agents=args.eval_agents,
        lower_priority=args.no_bg,
        bind_all=args.bind_all,
        load_experiment=args.load_experiment,
        output_value_heatmap=False,
        normalize_target_values=True,
        use_cached_values=False
    )

    trainer.run()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, interrupt_handler)
    signal.signal(signal.SIGTERM, interrupt_handler)

    parser = argparse.ArgumentParser(description='Run Deep RL for selfish mining in blockchain')
    parser.add_argument('--build_info', help='build identifier', default=None)
    parser.add_argument('--output_root', help='root destination for all output files', default=None)
    parser.add_argument('--train_agents', help='number of agents spawned to train', default=5, type=int)
    parser.add_argument('--eval_agents', help='number of agents spawned to evaluate', default=2, type=int)
    parser.add_argument('--no_bg', help='don\'t run the process in the background', action='store_false')
    parser.add_argument('--bind_all', help='pass bind_all to tensorboard', action='store_true')
    parser.add_argument('--load_experiment', help='name of experiment to continue', default=None)
    # parser.add_argument('--alpha', help='miner size', default=0.01, type=float)
    parser.add_argument('--alpha', help='honest rate', default=0.4, type=float)
    parser.add_argument('--beta', help='attacker rate', default=0.6, type=float)
    parser.add_argument('--gamma', help='rushing factor', default=0.95, type=float)
    parser.add_argument('--max_fork', help='maximal fork size', default=5, type=int)
    parser.add_argument('--fee', help='transaction fee', default=10, type=float)
    parser.add_argument('--delta', help='chance for a transaction', default=0.01, type=float)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--lr', help='learning_rate', default=2e-4, type=float)

    # solve_pt(parser.parse_args())
    run_mcts_fees(parser.parse_args())
