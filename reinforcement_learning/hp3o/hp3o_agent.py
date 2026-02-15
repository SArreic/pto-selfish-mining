from typing import Tuple
import torch
from torch.distributions import Categorical
from reinforcement_learning.base.experience_acquisition.agents.bva_agent import BVAAgent
from reinforcement_learning.base.experience_acquisition.experience import Experience

class HP3OAgent(BVAAgent):
    def __init__(self, approximator, simulator, use_cache=True):
        super().__init__(approximator, simulator, use_cache=use_cache)
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'done_list': [],
            'next_states': [],
            'legal_actions': [],
            'difficulty': [],
            'prev_difficulty': []
        }

    def evaluate_state(self, state: torch.Tensor, exploring: bool) -> torch.Tensor:
        out = self.approximator(state)
        # MLPApproximator returns value at index 0
        return out[..., 0:1]

    def plan_action(self, explore: bool = True) -> Tuple[int, torch.Tensor]:
        state_tensor = self.current_state
        out = self.approximator(state_tensor)
        
        # Split output: value head and policy (logits) head
        value = out[..., 0:1]
        logits = out[..., 1:]
        
        # Masking
        mask = self.legal_actions.clone() # Use clone to avoid modifying original if shared
        
        # --- PREVENTION OF EARLY EXIT ---
        # Prohibit Exit action (type 6) when state is effectively empty/initial
        state_tuple = self.simulator.torch_to_tuple(state_tensor)
        if len(state_tuple) >= 7:
            priv_len, pub_len, slashed = state_tuple[4], state_tuple[5], state_tuple[6]
            # PREVENTION OF EARLY EXIT: If we have no private lead and are not slashed, 
            # don't exit. This forces exploration of mining strategies.
            if priv_len <= pub_len and slashed == 0:
                action_space = self.simulator.action_space
                num_action_types = action_space.intervals[0].size
                exit_action_type = int(self.simulator._model.Action.Exit)
                
                # Mask all variants of the Exit action (for all param values)
                for param in range(self.simulator._model.max_fork + 1):
                    exit_idx = param * num_action_types + exit_action_type
                    if exit_idx < len(mask):
                        mask[exit_idx] = False
        
        logits = logits.masked_fill(~mask, float('-inf'))
        
        dist = Categorical(logits=logits)
        if explore:
            action = dist.sample()
        else:
            action = torch.argmax(logits)
            
        log_prob = dist.log_prob(action)
        
        # Store transition data for HP3O trajectory collection
        self.temp_transition = {
            'state': state_tensor.detach(),
            'action': action.detach(),
            'log_prob': log_prob.detach(),
            'value': value.detach().squeeze(),
            'legal_actions': self.legal_actions.detach()
        }
        
        return int(action.item()), value.detach()

    def choose_action(self, explore: bool = True) -> int:
        # PlanningAgent overrides this, but we implement it just in case
        return super().choose_action(explore)

    def step(self, explore=True) -> Experience:
        # Get experience from parent step
        exp = super().step(explore=explore)
        
        # Append to current trajectory
        self.current_trajectory['states'].append(self.temp_transition['state'])
        self.current_trajectory['actions'].append(self.temp_transition['action'])
        self.current_trajectory['log_probs'].append(self.temp_transition['log_prob'])
        self.current_trajectory['values'].append(self.temp_transition['value'])
        self.current_trajectory['rewards'].append(exp.reward)
        self.current_trajectory['done_list'].append(exp.is_done)
        self.current_trajectory['next_states'].append(exp.next_state.detach())
        self.current_trajectory['legal_actions'].append(self.temp_transition['legal_actions'])
        self.current_trajectory['difficulty'].append(exp.difficulty_contribution)
        self.current_trajectory['prev_difficulty'].append(exp.prev_difficulty_contribution)
        
        return exp

    def reset(self, state=None, keep_state=False) -> None:
        super().reset(state, keep_state)
        # Clear trajectory data for new episode
        self.current_trajectory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'done_list': [],
            'next_states': [],
            'legal_actions': [],
            'difficulty': [],
            'prev_difficulty': []
        }

    def get_full_trajectory(self):
        return self.current_trajectory

    def update(self, approximator=None, **kwargs) -> None:
        if approximator is not None:
             # Use the base update logic
             self.approximator.update(approximator)
             self.approximator.eval()
