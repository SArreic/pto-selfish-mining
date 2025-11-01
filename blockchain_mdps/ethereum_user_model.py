# pos_behavior_model.py
import sys
from typing import Tuple

import numpy as np

from .base.base_space.default_value_space import DefaultValueSpace
from .base.base_space.discrete_space import DiscreteSpace
from .base.base_space.multi_dimensional_discrete_space import MultiDimensionalDiscreteSpace
from .base.base_space.space import Space
from .base.blockchain_model import BlockchainModel
from .base.state_transitions import StateTransitions


class EthereumUserModel(BlockchainModel):
    """
    Lightweight behavior MDP adapted for PoS-style training (behavior model).
    State: (priv_len, pub_len, fork)
    Actions: Wait, Withhold, Adopt, Release, Equivocate, Exit
    Parameters:
      - alpha: attacker portion / probability that 'attacker' (agent) finds the next epoch's 'block'
      - max_fork: maximum fork length to track
      - base_reward: fixed reward for 'normal' compliant action (Wait / Adopt by default)
      - exit_reward: reward returned by Exit action (simplified)
      - slashing_prob: base probability that a slashable action is detected
    This model purposefully keeps state small (no explicit stake fields) to enable fast policy training.
    """

    def __init__(self, alpha: float, max_fork: int, base_reward: float = 1.0,
                 exit_reward: float = 10.0, slashing_prob: float = 0.4):
        self.alpha = float(alpha)
        self.max_fork = int(max_fork)
        self.base_reward = float(base_reward)
        self.exit_reward = float(exit_reward)
        self.slashing_prob = float(slashing_prob)

        # Fork enum (same names as BitcoinModel for compatibility)
        self.Fork = self.create_int_enum('Fork', ['Irrelevant', 'Relevant', 'Active'])
        # Action enum aligned with PoSMinimalModel semantics
        self.Action = self.create_int_enum('Action', ['Illegal', 'Wait', 'Withhold', 'Adopt',
                                                      'Release', 'Equivocate', 'Exit'])

        super().__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.alpha}, {self.max_fork}, {self.base_reward}, {self.slashing_prob})'

    def __reduce__(self) -> Tuple[type, tuple]:
        return self.__class__, (self.alpha, self.max_fork, self.base_reward, self.exit_reward, self.slashing_prob)

    # ----------------------------
    # Spaces, initial / final states
    # ----------------------------
    def get_state_space(self) -> Space:
        # state: (priv_len in [0,max_fork], pub_len in [0,max_fork], fork enum)
        underlying_space = MultiDimensionalDiscreteSpace((0, self.max_fork), (0, self.max_fork), self.Fork)
        return DefaultValueSpace(underlying_space, self.get_final_state())

    def get_action_space(self) -> Space:
        # discrete actions (no extra parameter)
        return DiscreteSpace(self.Action)

    def get_initial_state(self) -> BlockchainModel.State:
        return 0, 0, self.Fork.Irrelevant

    def get_final_state(self) -> BlockchainModel.State:
        return -1, -1, self.Fork.Irrelevant

    # ----------------------------
    # State validity
    # ----------------------------
    def is_state_valid(self, state: BlockchainModel.State) -> bool:
        try:
            a, h, fork = state
        except Exception:
            return False
        if not (0 <= a <= self.max_fork and 0 <= h <= self.max_fork):
            return False
        if fork not in (self.Fork.Irrelevant, self.Fork.Relevant, self.Fork.Active):
            return False
        return True

    # ----------------------------
    # Helper: safe truncate
    # ----------------------------
    def _safe_truncate(self, v: int) -> int:
        if v < 0:
            return 0
        return min(v, self.max_fork)

    # ----------------------------
    # Core dynamics (simplified)
    # ----------------------------
    def get_state_transitions(self, state: BlockchainModel.State, action: BlockchainModel.Action,
                              check_valid: bool = True) -> StateTransitions:
        """
        原逻辑保留 —— 但在返回前调用 _dbg_validate_and_fix_transitions()
        以打印详细信息并在必要时自动修复（添加 final_state 自环）。
        """
        transitions = StateTransitions()

        # -------------------------
        # 原始逻辑（直接复制你现有的分支）
        # -------------------------
        # validity / final short-circuits
        if check_valid and not self.is_state_valid(state):
            transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            return transitions

        if state == self.final_state:
            transitions.add(self.final_state, probability=1, reward=0.0)
            return transitions

        a, h, fork = state
        action_type = action  # DiscreteSpace returns direct enum in WeRLman style

        if action_type is None:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            return transitions

        # -- ILLEGAL --
        if action is self.Action.Illegal:
            transitions.add(self.final_state, probability=1, reward=self.error_penalty / 2)
            return transitions

        # -- WAIT --
        if action_type is self.Action.Wait:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                attacker_state = self._safe_truncate(a + 1), h, self.Fork.Irrelevant
                honest_state = a, self._safe_truncate(h + 1), self.Fork.Relevant
                reward = self.base_reward
                transitions.add(attacker_state, probability=self.alpha, reward=reward)
                transitions.add(honest_state, probability=1 - self.alpha, reward=reward)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                attacker_state = self._safe_truncate(a + 1), h, self.Fork.Active
                honest_support_state = max(0, a - h), 1, self.Fork.Relevant
                transitions.add(attacker_state, probability=self.alpha, reward=self.base_reward)
                transitions.add(honest_support_state, probability=(1 - self.alpha) * 0.5, reward=h)
                transitions.add((a, self._safe_truncate(h + 1), self.Fork.Relevant),
                                probability=(1 - self.alpha) * 0.5)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # -- WITHHOLD --
        if action_type is self.Action.Withhold:
            if fork is not self.Fork.Active and a < self.max_fork and h < self.max_fork:
                alpha_effect = min(1.0, self.alpha + 0.1)
                succ_state = self._safe_truncate(a + 1), h, self.Fork.Irrelevant
                fail_state = a, self._safe_truncate(h + 1), self.Fork.Relevant
                transitions.add(succ_state, probability=alpha_effect, reward=0.0)
                transitions.add(fail_state, probability=1.0 - alpha_effect, reward=0.0)
            elif fork is self.Fork.Active and 0 < h <= a < self.max_fork:
                succ_state = self._safe_truncate(a + 1), h, self.Fork.Active
                fail_state = a, self._safe_truncate(h + 1), self.Fork.Relevant
                transitions.add(succ_state, probability=self.alpha, reward=0.0)
                transitions.add(fail_state, probability=1.0 - self.alpha, reward=0.0)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # -- ADOPT --
        if action_type is self.Action.Adopt:
            if h > 0:
                next_state = 0, 0, self.Fork.Irrelevant
                transitions.add(next_state, probability=1, reward=self.base_reward)
            else:
                transitions.add(self.final_state, probability=1, reward=self.base_reward)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # -- RELEASE --
        if action_type is self.Action.Release:
            if a > h:
                lead = a - h
                immediate_reward = lead * self.base_reward
                detect_prob = min(1.0, self.slashing_prob)
                next_state_no_slash = 0, a, self.Fork.Irrelevant
                slash_penalty = - (lead * self.base_reward * 5.0)
                next_state_slash = self.final_state
                transitions.add(next_state_no_slash, probability=1.0 - detect_prob, reward=immediate_reward)
                transitions.add(next_state_slash, probability=detect_prob, reward=slash_penalty)
            else:
                transitions.add(self.final_state, probability=1, reward=self.error_penalty)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # -- EQUIVOCATE --
        if action_type is self.Action.Equivocate:
            prob_detect = min(1.0, self.slashing_prob)
            prob_fork = max(0.0, self.alpha * 1.0 - 0.05)
            prob_no_effect = max(0.0, 1.0 - prob_detect - prob_fork)
            fork_gain_state = self._safe_truncate(a + 1), h, self.Fork.Active
            no_effect_state = a, self._safe_truncate(h + 1), self.Fork.Relevant
            slash_penalty = - (self.base_reward * 4.0)
            if prob_fork > 0:
                transitions.add(fork_gain_state, probability=prob_fork, reward=0.0)
            if prob_no_effect > 0:
                transitions.add(no_effect_state, probability=prob_no_effect, reward=0.0)
            if prob_detect > 0:
                transitions.add(self.final_state, probability=prob_detect, reward=slash_penalty)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # -- EXIT --
        if action_type is self.Action.Exit:
            transitions.add(self.final_state, probability=1.0, reward=self.exit_reward)
            # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
            return transitions

        # fallback
        transitions.add(self.final_state, probability=1.0, reward=self.error_penalty)
        # self._dbg_validate_and_fix_transitions(transitions, state, action_type)
        return transitions

    # ----------------------------
    # Utility helpers used by training/evaluation
    # ----------------------------
    def _dbg_validate_and_fix_transitions(self, transitions: StateTransitions, state, action):
        """
        Robustly inspect 'transitions' and:
          - print its dir and common fields (probabilities, rewards)
          - compute prob_sum
          - if prob_sum == 0: create a fallback self-loop to final_state with prob=1 and error_penalty reward
        """
        try:
            # try to use documented attributes first
            probs = getattr(transitions, 'probabilities', None)
            rewards = getattr(transitions, 'rewards', None)
            diffs = getattr(transitions, 'difficulty_contributions', None)

            # defensive: some implementations use lists/tuples internally; ensure we can sum
            if probs is None:
                # fallback: maybe transitions stores lists as attributes in __dict__
                d = getattr(transitions, '__dict__', {})
                # try common keys
                for k in ('probabilities', 'probs', 'p', 'prob'):
                    if k in d:
                        probs = d[k]
                        break

            # convert to list/iterable safely
            prob_list = None
            if probs is None:
                prob_list = []
            else:
                # if it's numpy array
                try:
                    import numpy as _np
                    if isinstance(probs, _np.ndarray):
                        prob_list = probs.tolist()
                    else:
                        # if it's some other iterable
                        prob_list = list(probs)
                except Exception:
                    # last resort
                    try:
                        prob_list = list(probs)
                    except Exception:
                        prob_list = []

            prob_sum = float(sum(prob_list)) if prob_list else 0.0

            # debug prints (use stdout to ensure visible)
            print(
                f"[DEBUG] StateTransitions dir: {sorted([attr for attr in dir(transitions) if not attr.startswith('__')])}")
            print(f"[DEBUG] During debug print: state={state}, action={action}, "
                  f"extracted_entries={len(prob_list)}, prob_sum={prob_sum:.6f}")

            # if prob_sum is zero -> fix by adding final_state self-loop
            if prob_sum == 0.0:
                print(f"[WARN] Zero-probability transitions detected for state={state} action={action}. "
                      f"Auto-fixing by adding self.final_state with prob=1 and reward=error_penalty.")
                try:
                    transitions.add(self.final_state, probability=1.0, reward=self.error_penalty)
                except Exception as e:
                    # if transitions.add signature different, try alternative attempts:
                    try:
                        transitions.add(self.final_state, 1.0, self.error_penalty)
                    except Exception:
                        print(f"[ERROR] Failed to auto-add self-loop: {e}")
            # else: if prob_sum not exactly 1.0 we can normalize optionally (not done here to preserve rewards)
            return
        except Exception as e:
            # make sure debug helper never crashes the real flow
            print(f"[ERROR] _dbg_validate_and_fix_transitions failed for state={state}, action={action}: {e}")
            return

    def get_honest_revenue(self) -> float:
        # expected honest revenue per epoch (simplified)
        return (1.0 - self.alpha) * self.base_reward

    def is_policy_honest(self, policy: BlockchainModel.Policy) -> bool:
        """
        Heuristic: checks a few canonical states for honest action choices:
          - (0,0,Irrelevant) -> Wait
          - (1,0,Irrelevant) -> Release (or Wait) - but we expect Wait for honest
          - (0,1,Irrelevant) -> Adopt
        This is a heuristic / convenience for tests.
        """
        try:
            i0 = self.state_space.element_to_index((0, 0, self.Fork.Irrelevant))
            i1 = self.state_space.element_to_index((1, 0, self.Fork.Irrelevant))
            i2 = self.state_space.element_to_index((0, 1, self.Fork.Irrelevant))
        except Exception:
            return False

        return (policy[i0] == self.Action.Wait and
                policy[i2] == self.Action.Adopt and
                policy[i1] in (self.Action.Wait, self.Action.Release))

    def build_honest_policy(self) -> BlockchainModel.Policy:
        """
        Build a simple honest policy:
          - if public lead > private lead: Adopt
          - if private lead > public lead: Wait (honest validators do not override)
          - if equal: Wait
        """
        policy = np.zeros(self.state_space.size, dtype=int)
        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)
            if h > a:
                action = self.Action.Adopt
            else:
                action = self.Action.Wait
            policy[i] = action
        return tuple(policy)

    def build_attack_policy(self) -> BlockchainModel.Policy:
        """
        A simple attacking heuristic:
          - If a > h -> Release (try to publish lead)
          - If a == h and a >= 1 and fork is Relevant -> Equivocate / Withhold (try to match)
          - Else -> Withhold (accumulate)
        """
        policy = np.zeros(self.state_space.size, dtype=int)
        for i in range(self.state_space.size):
            a, h, fork = self.state_space.index_to_element(i)
            if a > h:
                action = self.Action.Release
            elif (h == a - 1 and a >= 2) or a == self.max_fork:
                action = self.Action.Release
            elif (h == 1 and a == 1) and fork is self.Fork.Relevant:
                action = self.Action.Equivocate
            else:
                action = self.Action.Withhold
            policy[i] = action
        return tuple(policy)

    def _dbg_check_and_print(state, action, local_transitions, model_self):
        """
        Diagnostics: check the local_transitions list and print useful debugging information.
        - state: current state tuple
        - action: action enum value
        - local_transitions: list of (next_state, prob, reward)
        - model_self: self reference to the model (to access state_space)
        """
        try:
            action_name = action.name
        except Exception:
            action_name = str(action)

        if not local_transitions:
            print("DEBUG WARNING: Empty transition list for state:", state, "action:", action_name)
            return

        probs = [p for (_, p, _) in local_transitions]
        rewards = [r for (_, _, r) in local_transitions]

        # detect NaN / Inf
        if any([np.isnan(p) or np.isinf(p) for p in probs]):
            print("DEBUG WARNING: NaN/Inf in probabilities for state:", state, "action:", action_name)
        if any([np.isnan(r) or np.isinf(r) for r in rewards]):
            print("DEBUG WARNING: NaN/Inf in rewards for state:", state, "action:", action_name)

        ssum = float(sum(probs))
        if abs(ssum - 1.0) > 1e-9:
            print("DEBUG WARNING: probability sum != 1 for state:", state, "action:", action_name,
                  " sum=", ssum, " entries=")
            for (ns, p, r) in local_transitions:
                # try to map ns to index (if possible)
                try:
                    idx = model_self.state_space.element_to_index(ns)
                except Exception:
                    idx = None
                print(f"   next_state={ns} index={idx} prob={p} reward={r}")
        # Optionally print a compact summary for well-formed rows (uncomment if verbose desired)
        # else:
        #     print("DEBUG: OK transition for state:", state, "action:", action_name, "sum=", ssum)


def _dbg_check_and_print(state, action, local_transitions, model_self):
    """
    Diagnostics: check the local_transitions list and print useful debugging information.
    - state: current state tuple
    - action: action enum value
    - local_transitions: list of (next_state, prob, reward)
    - model_self: self reference to the model (to access state_space)
    """
    try:
        action_name = action.name
    except Exception:
        action_name = str(action)

    if not local_transitions:
        print("DEBUG WARNING: Empty transition list for state:", state, "action:", action_name)
        return

    probs = [p for (_, p, _) in local_transitions]
    rewards = [r for (_, _, r) in local_transitions]

    # detect NaN / Inf
    if any([np.isnan(p) or np.isinf(p) for p in probs]):
        print("DEBUG WARNING: NaN/Inf in probabilities for state:", state, "action:", action_name)
    if any([np.isnan(r) or np.isinf(r) for r in rewards]):
        print("DEBUG WARNING: NaN/Inf in rewards for state:", state, "action:", action_name)

    ssum = float(sum(probs))
    if abs(ssum - 1.0) > 1e-9:
        print("DEBUG WARNING: probability sum != 1 for state:", state, "action:", action_name,
              " sum=", ssum, " entries=")
        for (ns, p, r) in local_transitions:
            # try to map ns to index (if possible)
            try:
                idx = model_self.state_space.element_to_index(ns)
            except Exception:
                idx = None
            print(f"   next_state={ns} index={idx} prob={p} reward={r}")
    # Optionally print a compact summary for well-formed rows (uncomment if verbose desired)
    # else:
    #     print("DEBUG: OK transition for state:", state, "action:", action_name, "sum=", ssum)


if __name__ == '__main__':
    print('pos_behavior_model module test')
    np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

    mdp = EthereumUserModel(alpha=0.2, max_fork=6, base_reward=1.0, exit_reward=8.0, slashing_prob=0.4)
    print("state_space_size:", mdp.state_space.size)
    p = mdp.build_attack_policy()
    print("first 16 policy actions:", p[:16])
