# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-04-14 00:47:29
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-04-14 00:47:29


from collections import deque
from .hmm_model import HMModel
from .payoff_matrix import get_optimal_response
from .constants import NEGATIVE_EMOTIONS

class EmotionInteractionSystem:
    def __init__(self):
        self.customer_history = deque(maxlen=5)
        self.robot_history = deque(maxlen=5)
        self.hmm = HMModel()
    
    def get_next_emotion(self, customer_emotion, current_robot_emotion=None):
        self.customer_history.append(customer_emotion)
        
        prev_robot_emotion = self.robot_history[-1] if self.robot_history else None
        
        if current_robot_emotion is None:
            next_emotion = get_optimal_response(customer_emotion)
        elif self._should_use_hmm():
            next_emotion = self._get_hmm_response(customer_emotion, current_robot_emotion)
        else:
            next_emotion = get_optimal_response(customer_emotion)
        
        self._update_learning(prev_robot_emotion, current_robot_emotion, customer_emotion)
        self.robot_history.append(next_emotion)
        return next_emotion
    
    def _should_use_hmm(self):
        if len(self.customer_history) < 5:
            return False
        return sum(e in NEGATIVE_EMOTIONS for e in self.customer_history) >= 3
    
    def _get_hmm_response(self, customer_emotion, current_robot_emotion):
        return self.hmm.predict_next_state(
            current_robot_emotion,
            customer_emotion,
            apply_negative_bias=True
        )
    
    def _update_learning(self, prev_state, current_state, observation):
        if prev_state and current_state:
            self.hmm.update(prev_state, current_state, observation)
