"""
Win-Stay, Lose-Shift (WSLS) Emotion Selection Strategy
Based on Section 3.2 and Algorithm 3 from the paper.

Eq. 1: f_Payoff(d) = argmax_{e in E} pi(d,e)_2
"""

import numpy as np
from typing import Optional
from models.constants import (
    EMOTIONS, N_EMOTIONS, EMOTION_TO_IDX, IDX_TO_EMOTION,
    PAYOFF_MATRIX_AGENT, PAYOFF_THRESHOLD
)


class WSLSStrategy:
    """
    Win-Stay, Lose-Shift emotion selection strategy.

    For positive debtor emotions, the creditor maintains cooperation.
    For negative exchanges, it shifts to cautious responses.
    Uses the payoff matrix to select optimal creditor emotion.
    """

    def __init__(self, payoff_threshold: float = PAYOFF_THRESHOLD):
        self.payoff_matrix = PAYOFF_MATRIX_AGENT.copy()
        self.payoff_threshold = payoff_threshold
        self.previous_emotion = None
        self.previous_payoff = None

    def select_emotion(self, debtor_emotion: str) -> str:
        """
        Algorithm 3: WSLS Emotion Selection

        Eq. 1: f_Payoff(d) = argmax_{e in E} pi(d,e)_2

        Args:
            debtor_emotion: Current debtor emotion string

        Returns:
            Selected creditor emotion string
        """
        d_idx = EMOTION_TO_IDX.get(debtor_emotion, EMOTION_TO_IDX['Neutral'])

        # Calculate payoffs for all possible agent emotions
        payoffs = self.payoff_matrix[d_idx]

        # Select emotion with highest agent payoff
        best_idx = np.argmax(payoffs)
        best_emotion = IDX_TO_EMOTION[best_idx]

        # Win-Stay, Lose-Shift logic
        if self.previous_emotion is not None and self.previous_payoff is not None:
            if self.previous_payoff < self.payoff_threshold:
                # Lose condition: shift to second-best emotion
                sorted_indices = np.argsort(payoffs)[::-1]
                if len(sorted_indices) > 1:
                    best_emotion = IDX_TO_EMOTION[sorted_indices[1]]
                    best_idx = sorted_indices[1]

        # Update tracking
        self.previous_emotion = best_emotion
        self.previous_payoff = float(payoffs[best_idx])

        return best_emotion

    def get_payoff(self, debtor_emotion: str, agent_emotion: str) -> float:
        """Get the agent payoff for a given emotion pair."""
        d_idx = EMOTION_TO_IDX.get(debtor_emotion, EMOTION_TO_IDX['Neutral'])
        a_idx = EMOTION_TO_IDX.get(agent_emotion, EMOTION_TO_IDX['Neutral'])
        return float(self.payoff_matrix[d_idx, a_idx])

    def update_payoff_matrix(self, debtor_emotion: str, agent_emotion: str,
                             reward_delta: float, learning_rate: float = 0.1):
        """Online learning: update payoff matrix based on negotiation outcome."""
        d_idx = EMOTION_TO_IDX.get(debtor_emotion, EMOTION_TO_IDX['Neutral'])
        a_idx = EMOTION_TO_IDX.get(agent_emotion, EMOTION_TO_IDX['Neutral'])
        self.payoff_matrix[d_idx, a_idx] += learning_rate * reward_delta

    def reset(self):
        """Reset for new negotiation."""
        self.previous_emotion = None
        self.previous_payoff = None
