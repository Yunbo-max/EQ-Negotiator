"""
EQ-Negotiator: Main emotion strategy engine.
Combines WSLS payoff strategy with HMM-based emotional adaptation (Eq. 4).

Policy selection:
  C_{t+1} = f_HMM(...)  if negative emotion count >= k in last n turns
  C_{t+1} = f_Payoff(D_t) otherwise
"""

from typing import Dict, List, Any
from models.hmm_model import HMMEmotionModel
from models.payoff_matrix import WSLSStrategy
from models.constants import (
    EMOTIONS, EMOTION_TO_IDX, EMOTION_PROMPTS,
    NEGATIVE_EMOTIONS, HMM_ACTIVATION_K, HMM_ACTIVATION_N
)


class EQNegotiator:
    """
    EQ-Negotiator emotion engine.
    Integrates game-theoretic WSLS with HMM for dynamic emotional adaptation.
    """

    def __init__(self, activation_k: int = HMM_ACTIVATION_K,
                 activation_n: int = HMM_ACTIVATION_N):
        self.hmm = HMMEmotionModel(activation_k, activation_n)
        self.wsls = WSLSStrategy()

        self.current_emotion = 'Neutral'
        self.emotion_sequence: List[str] = []
        self.strategy_log: List[str] = []  # Track which strategy was used

    def select_emotion(self, debtor_emotion: str) -> Dict[str, Any]:
        """
        Select next creditor emotion based on Eq. 4:
        - Use HMM if negative emotion persistence detected
        - Use WSLS payoff otherwise

        Args:
            debtor_emotion: Detected debtor emotion

        Returns:
            Dict with emotion config for the LLM prompt
        """
        # Record observation
        self.hmm.add_observation(debtor_emotion, self.current_emotion)

        # Policy selection (Eq. 4)
        if self.hmm.should_activate():
            # HMM-based selection (Eq. 5)
            next_emotion = self.hmm.select_emotion(debtor_emotion, self.current_emotion)
            strategy_used = 'HMM'
        else:
            # WSLS payoff selection (Eq. 1)
            next_emotion = self.wsls.select_emotion(debtor_emotion)
            strategy_used = 'WSLS'

        self.current_emotion = next_emotion
        self.emotion_sequence.append(next_emotion)
        self.strategy_log.append(strategy_used)

        # Temperature decay: tau(t) = max(0.1, 0.7 * (1-0.05)^t)
        t = len(self.emotion_sequence)
        temperature = max(0.1, 0.7 * (0.95 ** t))

        return {
            'emotion': next_emotion,
            'emotion_text': EMOTION_PROMPTS.get(next_emotion, EMOTION_PROMPTS['Neutral']),
            'temperature': temperature,
            'strategy': strategy_used,
            'hmm_active': self.hmm.should_activate(),
            'belief_state': self.hmm.get_belief_state(),
            'dominant_state': self.hmm.get_dominant_state(),
        }

    def update_after_negotiation(self, result: Dict[str, Any]):
        """Update models after a complete negotiation."""
        success = result.get('final_state') == 'accept'

        # Update WSLS payoff based on outcome
        if self.emotion_sequence and self.hmm.debtor_history:
            reward_delta = 1.0 if success else -0.5
            for d_emo, c_emo in zip(self.hmm.debtor_history, self.emotion_sequence):
                self.wsls.update_payoff_matrix(d_emo, c_emo, reward_delta, learning_rate=0.05)

        # Update HMM parameters from sequences
        if len(self.hmm.debtor_history) > 1:
            self.hmm.update_parameters(
                [self.hmm.debtor_history],
                [self.emotion_sequence],
                learning_rate=0.03
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        hmm_stats = self.hmm.get_stats()
        strategy_counts = {}
        for s in self.strategy_log:
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        return {
            **hmm_stats,
            'current_emotion': self.current_emotion,
            'emotion_sequence_length': len(self.emotion_sequence),
            'strategy_counts': strategy_counts,
            'emotion_sequence': self.emotion_sequence[-10:],
        }

    def reset(self):
        """Reset for new negotiation (keep learned parameters)."""
        self.current_emotion = 'Neutral'
        self.emotion_sequence = []
        self.strategy_log = []
        self.hmm.reset()
        self.wsls.reset()
