# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-04-14 00:47:16
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-04-14 00:47:16


import numpy as np
from .constants import EMOTIONS, DEFAULT_TRANSITION_MATRIX, DEFAULT_EMISSION_MATRIX, DEFAULT_NEGATIVE_BIAS

class HMModel:
    def __init__(self):
        self.transition_counts = np.array(DEFAULT_TRANSITION_MATRIX)
        self.emission_counts = np.array(DEFAULT_EMISSION_MATRIX)
        self.negative_bias = np.array(DEFAULT_NEGATIVE_BIAS)
        self._update_matrices()
    
    def _update_matrices(self):
        self.transition_matrix = self.transition_counts / self.transition_counts.sum(axis=1, keepdims=True)
        self.emission_matrix = self.emission_counts / self.emission_counts.sum(axis=1, keepdims=True)
    
    def update(self, prev_state, current_state, observation):
        prev_idx = EMOTIONS.index(prev_state)
        current_idx = EMOTIONS.index(current_state)
        obs_idx = EMOTIONS.index(observation)
        
        self.transition_counts[prev_idx, current_idx] += 1
        self.emission_counts[current_idx, obs_idx] += 1
        self._update_matrices()
    
    def predict_next_state(self, current_state, observation, apply_negative_bias=False):
        current_idx = EMOTIONS.index(current_state)
        obs_idx = EMOTIONS.index(observation)
        
        state_probs = (self.transition_matrix[current_idx] * 
                       self.emission_matrix[:, obs_idx])
        
        if apply_negative_bias:
            state_probs *= self.negative_bias
        
        state_probs /= state_probs.sum()
        next_state_idx = np.argmax(state_probs)
        return EMOTIONS[next_state_idx]