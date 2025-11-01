# pos_mdp.py
import sys
from enum import Enum
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class EthereumPoSModel(BlockchainModel):
    """
    Minimal PoS-style MDP model for epoch-level decisions.
    State tuple layout (final version):
      (pub_chain_repr, priv_chain_repr, fork_flag, stake_pool, stake_user, priv_len, pub_len, slashed_flag)
    For simplicity we represent chains only via lengths (priv_len, pub_len) plus fork_flag.
    """

    def __init__(self,
                 alpha: float,
                 max_fork: int = 10,
                 base_reward: float = 1.0,
                 slashing_base_prob: float = 0.5,
                 exit_immediate: bool = True):
        """
        :param alpha: attacker fraction in network (used for some transition probabilities)
        :param max_fork: maximum fork length to track (truncate/limit)
        :param base_reward: fixed reward per epoch for compliant wait behavior
        :param slashing_base_prob: baseline probability that a slashable action is detected (can be scaled)
        :param exit_immediate: if True, Exit immediately returns stake_user to agent and ends participation
        """
        self.alpha = float(alpha)
        self.max_fork = int(max_fork)
        self.base_reward = float(base_reward)
        self.slashing_base_prob = float(slashing_base_prob)
        self.exit_immediate = bool(exit_immediate)

        # define enums like BitcoinFeeModel style
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        # self.Action = self.create_int_enum('Action', ['Wait', 'Withhold', 'Adopt', 'Release', 'Equivocate', 'Exit'])
        self.Action = self.create_int_enum('Action', ['Illegal', 'Wait', 'Withhold', 'Adopt', 'Release', 'Equivocate'])
        # Make minimal placeholders for chain block representation (not used as vector here)
        self.Dummy = self.create_int_enum('Dummy', ['Zero'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork}, {self.base_reward}, {self.slashing_base_prob})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (
            self.alpha, self.max_fork, self.base_reward, self.slashing_base_prob, self.exit_immediate)

    # -----------------------
    # State space definitions
    # We'll use a compact tuple:
    # (dummy_chain_repr) * 1  + (Fork,) + (0..stake_pool_max) +
    # (0..stake_pool_max) + (0..max_fork) + (0..max_fork) + (0..1)
    # For compatibility with existing code's MultiDimensionalDiscreteSpace we create elements accordingly.
    # -----------------------
    def get_state_space(self) -> Space:
        # Use dummy for compatibility (keeps shape similar to bitcoin model)
        elements = [self.Dummy] \
                   + [self.Fork, (0, 1000000), (0, 1000000), (0, self.max_fork), (0, self.max_fork), (0, 1)]
        # meaning: Dummy, Fork, stake_pool, stake_user, priv_len, pub_len, slashed_flag
        underlying_space = MultiDimensionalDiscreteSpace(*elements)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        # Action param is unused for most; keep (0, max_fork) to allow Release(k) if desired later.
        return MultiDimensionalDiscreteSpace(self.Action, (0, self.max_fork))

    def get_initial_state(self) -> BlockchainModel.State:
        # initial: dummy=Zero, Fork.Irrelevant, stake_pool=100 (example), stake_user=1 (example), priv_len=0,
        # pub_len=0, slashed_flag=0
        init_pool = 100
        init_user = 1
        return self.Dummy.Zero, self.Fork.Irrelevant, init_pool, init_user, 0, 0, 0

    def get_final_state(self) -> BlockchainModel.State:
        return self.Dummy.Zero, self.Fork.Irrelevant, -1, -1, -1, -1, -1

    def dissect_state(self, state: BlockchainModel.State):
        dummy = state[0]
        fork = state[1]
        pool = int(state[2])
        user = int(state[3])
        priv_len = int(state[4])
        pub_len = int(state[5])
        slashed = int(state[6])
        return dummy, fork, pool, user, priv_len, pub_len, slashed

    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        try:
            dummy, fork, pool, user, priv_len, pub_len, slashed = self.dissect_state(state)
        except Exception:
            return False
        # simple bounds check
        if not (0 <= pool):
            return False
        if not (0 <= user <= pool):
            return False
        if not (0 <= priv_len <= self.max_fork and 0 <= pub_len <= self.max_fork):
            return False
        if slashed not in (0, 1):
            return False
        return True

    # -----------------------
    # Helpers
    # -----------------------
    def final_state(self):
        return self.get_final_state()

    def safe_truncate(self, val: int) -> int:
        if val < 0:
            return 0
        return min(val, self.max_fork)

    # -----------------------
    # Transition dynamics (simplified)
    # -----------------------
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        transitions = StateTransitions()

        if check_valid and not self.is_state_valid(state):
            transitions.add(self.get_final_state(), probability=1, reward=self.error_penalty)
            return transitions

        if state == self.get_final_state():
            transitions.add(self.get_final_state(), probability=1)
            return transitions

        dummy, fork, pool, user, priv_len, pub_len, slashed = self.dissect_state(state)
        action_type, action_param = action

        # If already slashed then we treat that as still in-system but penalized:
        if slashed == 1:
            # after being slashed, agent continues but gets no base reward until exit (simplification).
            # For now, allow normal transitions but with zero base reward; could be extended.
            pass

        # ------- ILLEGAL -------
        if action_type is self.Action.Illegal:
            transitions.add(self.get_final_state(), probability=1, reward=self.error_penalty / 2)
            return transitions

        # ------- WAIT -------
        if action_type is self.Action.Wait:
            # Honest participants extend public chain with probability proportional to (1 - alpha)
            # Attacker may also extend its private chain proportional to alpha * stake ratio.
            stake_ratio = user / pool if pool > 0 else 0.0
            # Probability honest finds the next effective block
            p_honest = (1.0 - self.alpha)
            p_attacker = self.alpha

            # Two possible next states: attacker extends (priv_len+1) OR honest extends (pub_len+1)
            attacker_state = (
                dummy, self.Fork.Irrelevant, pool, user, self.safe_truncate(priv_len + 1), pub_len, slashed)
            honest_state = (dummy, self.Fork.Irrelevant, pool, user, priv_len, self.safe_truncate(pub_len + 1), slashed)

            # Reward: base_reward * stake_ratio for Wait (user participates and collects proportional reward)
            reward = self.base_reward * stake_ratio

            transitions.add(attacker_state, probability=p_attacker, reward=reward)
            transitions.add(honest_state, probability=p_honest, reward=reward)

            return transitions

        # ------- WITHHOLD -------
        if action_type is self.Action.Withhold:
            # Withhold: agent does not collect base reward this epoch; instead increases private lead with some
            # probability attacker_success depends on alpha and stake_ratio (if user is attacker)
            stake_ratio = user / pool if pool > 0 else 0.0
            p_success = min(1.0, self.alpha + stake_ratio * 0.5)  # heuristic prob of creating private block
            new_priv = self.safe_truncate(priv_len + 1)
            next_state_success = (dummy, self.Fork.Irrelevant, pool, user, new_priv, pub_len, slashed)
            # if fail (honest block), public increases
            next_state_fail = (
                dummy, self.Fork.Irrelevant, pool, user, priv_len, self.safe_truncate(pub_len + 1), slashed)

            # Withhold gives zero immediate reward
            transitions.add(next_state_success, probability=p_success, reward=0.0)
            transitions.add(next_state_fail, probability=1.0 - p_success, reward=0.0)

            return transitions

        # ------- ADOPT -------
        if action_type is self.Action.Adopt:
            # Adopt: abandon private chain, reset priv_len=pub_len=0, continue from public chain
            # No immediate reward but resets private lead
            # adopting empty private chain is allowed but trivial
            stake_ratio = user / pool if pool > 0 else 0.0
            reward = self.base_reward * stake_ratio
            next_state = (dummy, self.Fork.Irrelevant, pool, user, 0, 0, slashed)
            transitions.add(next_state, probability=1.0, reward=reward)
            return transitions

        # ------- RELEASE (reveal all private blocks) -------
        if action_type is self.Action.Release:
            # Release all private-leading blocks: if priv_len > pub_len, publisher may replace public chain
            if priv_len <= pub_len:
                # nothing to release (invalid release attempt)
                transitions.add(self.get_final_state(), probability=1.0, reward=self.error_penalty)
                return transitions

            lead = priv_len - pub_len
            # immediate reward: proportional to lead * base_reward * stake_ratio
            stake_ratio = user / pool if pool > 0 else 0.0
            immediate_reward = lead * self.base_reward * stake_ratio

            # risk of slashing: depends on slashing_base_prob scaled by equivocation likelihood
            # We model detection prob as slashing_base_prob * (1 - stake_ratio) (if attacker small -> easier to detect),
            # but capped
            detect_prob = min(1.0, self.slashing_base_prob * (1.0 - min(0.9, stake_ratio)))
            # if slashed -> big negative penalty and set slashed flag
            slashed_penalty = - (user * 0.5)  # heuristic large penalty; can be tuned or parameterized

            # Next states:
            # - success_no_slash: private chain becomes public; reset priv_len=0 and pub_len becomes previous priv_len
            next_state_no_slash = (dummy, self.Fork.Irrelevant, pool, user + immediate_reward, 0, priv_len, 0)
            # - detected_and_slashed: slashing happens; penalize and mark slashed flag, user loses stake proportionally
            user_after_slash = max(0, user - int(user * 0.5))  # sample penalty: 50% of user's stake removed
            pool_after_slash = max(0, pool - (user - user_after_slash))
            next_state_slashed = (dummy, self.Fork.Irrelevant, pool_after_slash, user_after_slash, 0, pub_len, 1)

            transitions.add(next_state_no_slash, probability=1.0 - detect_prob, reward=immediate_reward)
            transitions.add(next_state_slashed, probability=detect_prob, reward=slashed_penalty)

            return transitions

        # ------- EQUIVOCATE -------
        if action_type is self.Action.Equivocate:
            # Equivocate: create conflicting votes/signatures -> may cause fork or be detected
            # We model two outcomes: fork (benefit: increase priv_len or reduce pub_len) or detection->slashing
            stake_ratio = user / pool if pool > 0 else 0.0
            prob_fork = min(1.0, self.alpha + stake_ratio * 0.2)  # higher alpha -> more likely to cause fork
            detect_prob = min(1.0, self.slashing_base_prob * (1.0 - stake_ratio * 0.5))

            # Fork outcome: random effect - with some chance reduce pub_len by 1 or increase priv_len
            # We'll model fork as making private advantage +1 with probability 0.7 otherwise makes public stuck
            fork_gain_prob = 0.7
            state_on_fork_gain = (
                dummy, self.Fork.Active, pool, user, self.safe_truncate(priv_len + 1), pub_len, slashed)
            state_on_fork_no_gain = (dummy, self.Fork.Relevant, pool, user, priv_len, pub_len, slashed)

            # Detected -> heavy penalty and mark slashed
            user_after_slash = max(0, user - int(user * 0.5))
            pool_after_slash = max(0, pool - (user - user_after_slash))
            state_on_slash = (dummy, self.Fork.Irrelevant, pool_after_slash, user_after_slash, 0, pub_len, 1)
            slash_penalty = - (pool * 0.3)

            # Compose probabilities:
            p_fork = prob_fork * (1.0 - detect_prob)
            p_slash = detect_prob
            p_no_effect = 1.0 - p_fork - p_slash
            # distribute p_fork to gain/no_gain
            p_gain = p_fork * fork_gain_prob
            p_no_gain_fork = p_fork * (1.0 - fork_gain_prob)

            transitions.add(state_on_fork_gain, probability=p_gain, reward=0.0)
            transitions.add(state_on_fork_no_gain, probability=p_no_gain_fork, reward=0.0)
            transitions.add(state_on_slash, probability=p_slash, reward=slash_penalty)
            # If p_no_effect > 0, we model it as trivial no-op that advances public chain
            if p_no_effect > 0:
                next_pub_advance = (
                    dummy, self.Fork.Irrelevant, pool, user, priv_len, self.safe_truncate(pub_len + 1), slashed)
                transitions.add(next_pub_advance, probability=p_no_effect, reward=0.0)

            return transitions

        # ------- EXIT -------
        # if action_type is self.Action.Exit:
        #     # Exit: user collects stake_user and stops participating. For simplicity, go to final state.
        #     # reward: return stake_user (or proportion)
        #     reward = 0.0
        #     # final state representation uses get_final_state()
        #     transitions.add(self.get_final_state(), probability=1.0, reward=reward)
        #     return transitions

        # Fallback
        transitions.add(self.get_final_state(), probability=1.0, reward=self.error_penalty)
        return transitions

    def get_honest_revenue(self) -> float:
        # approximate expected honest revenue per epoch per unit stake
        return (1.0 - self.alpha) * self.base_reward


if __name__ == '__main__':
    print('pos_mdp module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = EthereumPoSModel(alpha=0.2, max_fork=5, base_reward=1.0, slashing_base_prob=0.4)
    print("state_space_size:", mdp.state_space.size)
    s0 = mdp.get_initial_state()
    print("initial:", s0)
    # sample transitions from Wait action
    act_wait = (mdp.Action.Wait, 0)
    trans = mdp.get_state_transitions(s0, act_wait)
    print("Wait transitions count:", len(trans))
    for ns, prob, rew, *_ in trans:
        print("next:", ns, "prob:", prob, "reward:", rew)
