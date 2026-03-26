"""
Hidden Markov Model for Emotional Strategy Selection
Based on Section 3.2 of the paper.

Implements Bayesian filtering (Eq. 2-3) and HMM-based emotion selection (Eq. 5-6).
Hidden states: Cooperative, Confrontational, Distressed, Strategic
Observable: (debtor_emotion, creditor_emotion) pairs
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from models.constants import (
    EMOTIONS, N_EMOTIONS, EMOTION_TO_IDX, IDX_TO_EMOTION,
    HIDDEN_STATES, N_STATES, STATE_TO_IDX,
    TRANSITION_MATRIX, EMISSION_MATRIX,
    STATE_TRANSITION, INITIAL_STATE_DIST, STATE_EMOTION_WEIGHTS,
    NEGATIVE_EMOTIONS, HMM_ACTIVATION_K, HMM_ACTIVATION_N
)


class HMMEmotionModel:
    """
    HMM-based emotional strategy model.

    Hidden states S_t in {Cooperative, Confrontational, Distressed, Strategic}
    Observable: emotion pairs (D_t, C_t) at each turn.

    Uses Bayesian filtering to maintain belief state bel(S_t) and
    selects optimal creditor emotion by maximizing expected utility.
    """

    def __init__(self, activation_k: int = HMM_ACTIVATION_K,
                 activation_n: int = HMM_ACTIVATION_N):
        # HMM parameters (theta)
        self.state_transition = STATE_TRANSITION.copy()      # P(S_{t+1} | S_t)
        self.emission_matrix = EMISSION_MATRIX.copy()        # P(D_t, C_t | S_t)
        self.transition_matrix = TRANSITION_MATRIX.copy()    # Agent emotion transitions

        # Belief state: P(S_t | D_{1:t}, C_{1:t})
        self.belief = INITIAL_STATE_DIST.copy()

        # State-emotion weight mapping
        self.state_emotion_weights = {}
        for state, weights in STATE_EMOTION_WEIGHTS.items():
            state_weights = np.zeros(N_EMOTIONS)
            for emotion, w in weights.items():
                state_weights[EMOTION_TO_IDX[emotion]] = w
            self.state_emotion_weights[STATE_TO_IDX[state]] = state_weights

        # Activation threshold
        self.activation_k = activation_k
        self.activation_n = activation_n

        # Emotion histories
        self.debtor_history: List[str] = []
        self.creditor_history: List[str] = []

        # Learning statistics
        self.total_updates = 0

    def should_activate(self) -> bool:
        """
        Check if HMM should activate (Eq. 4 condition).
        Activates when >= k negative emotions in last n turns.
        """
        if len(self.debtor_history) < self.activation_n:
            recent = self.debtor_history
        else:
            recent = self.debtor_history[-self.activation_n:]

        neg_count = sum(1 for e in recent if e in NEGATIVE_EMOTIONS)
        return neg_count >= self.activation_k

    def update_belief(self, debtor_emotion: str, creditor_emotion: str):
        """
        Bayesian filtering update (Eq. 2-3).

        bel(S_t) = eta * P(D_t, C_t | S_t) * sum_{S_{t-1}} P(S_t | S_{t-1}) * bel(S_{t-1})
        """
        d_idx = EMOTION_TO_IDX.get(debtor_emotion, EMOTION_TO_IDX['Neutral'])
        c_idx = EMOTION_TO_IDX.get(creditor_emotion, EMOTION_TO_IDX['Neutral'])

        # Compute emission likelihood P(D_t, C_t | S_t) for each hidden state
        emission_likelihood = np.zeros(N_STATES)
        for s in range(N_STATES):
            # Use state-emotion weights as proxy for emission
            state_weights = self.state_emotion_weights[s]
            # Joint probability approximated as product of marginals
            emission_likelihood[s] = (
                state_weights[c_idx] * self.emission_matrix[c_idx, d_idx]
            )

        # Prediction step: sum_{S_{t-1}} P(S_t | S_{t-1}) * bel(S_{t-1})
        predicted = self.state_transition.T @ self.belief

        # Update step: element-wise multiply with emission
        unnormalized = emission_likelihood * predicted

        # Normalization (Eq. 3)
        eta = np.sum(unnormalized)
        if eta > 0:
            self.belief = unnormalized / eta
        else:
            self.belief = np.ones(N_STATES) / N_STATES

        self.total_updates += 1

    def select_emotion(self, debtor_emotion: str, creditor_emotion: str) -> str:
        """
        HMM-based emotion selection (Eq. 5).

        f_HMM = argmax_{e in E} sum_{S_{t+1}} P(S_{t+1}|H) * P(D_{t+1}|S_{t+1},C_t=e) * w(e,D_t,S_{t+1})

        Args:
            debtor_emotion: Current debtor emotion
            creditor_emotion: Current creditor emotion

        Returns:
            Optimal next creditor emotion
        """
        d_idx = EMOTION_TO_IDX.get(debtor_emotion, EMOTION_TO_IDX['Neutral'])

        # Predict next hidden state distribution (Eq. 6)
        # P(S_{t+1} | H) = sum_{S_t} P(S_{t+1} | S_t) * bel(S_t)
        next_state_dist = self.state_transition.T @ self.belief

        # Evaluate each candidate creditor emotion
        emotion_scores = np.zeros(N_EMOTIONS)

        for e_idx in range(N_EMOTIONS):
            score = 0.0
            for s in range(N_STATES):
                # P(S_{t+1} | H)
                state_prob = next_state_dist[s]

                # P(D_{t+1} | S_{t+1}, C_t=e) - likelihood of positive debtor response
                # Use emission matrix as proxy
                debtor_response_prob = self.emission_matrix[e_idx, :].max()

                # w(e, D_t, S_{t+1}) - weight function combining strategic value
                state_weights = self.state_emotion_weights[s]
                weight = state_weights[e_idx]

                score += state_prob * debtor_response_prob * weight

            emotion_scores[e_idx] = score

        best_idx = np.argmax(emotion_scores)
        return IDX_TO_EMOTION[best_idx]

    def add_observation(self, debtor_emotion: str, creditor_emotion: str):
        """Record an observation and update belief state."""
        self.debtor_history.append(debtor_emotion)
        self.creditor_history.append(creditor_emotion)
        self.update_belief(debtor_emotion, creditor_emotion)

    def get_dominant_state(self) -> str:
        """Get the most likely hidden state."""
        idx = np.argmax(self.belief)
        return HIDDEN_STATES[idx]

    def get_belief_state(self) -> Dict[str, float]:
        """Get current belief distribution over hidden states."""
        return {HIDDEN_STATES[i]: float(self.belief[i]) for i in range(N_STATES)}

    def update_parameters(self, debtor_sequences: List[List[str]],
                          creditor_sequences: List[List[str]],
                          learning_rate: float = 0.05):
        """
        Online parameter learning (Eq. 7).
        Update transition and emission matrices from observed sequences.
        """
        for d_seq, c_seq in zip(debtor_sequences, creditor_sequences):
            for t in range(1, len(d_seq)):
                d_prev = EMOTION_TO_IDX.get(d_seq[t-1], 6)
                d_curr = EMOTION_TO_IDX.get(d_seq[t], 6)
                c_prev = EMOTION_TO_IDX.get(c_seq[t-1], 6)
                c_curr = EMOTION_TO_IDX.get(c_seq[t], 6)

                # Update emission matrix
                self.emission_matrix[c_curr, d_curr] += learning_rate
                # Re-normalize row
                row_sum = self.emission_matrix[c_curr].sum()
                if row_sum > 0:
                    self.emission_matrix[c_curr] /= row_sum

                # Update transition matrix
                self.transition_matrix[c_prev, c_curr] += learning_rate
                row_sum = self.transition_matrix[c_prev].sum()
                if row_sum > 0:
                    self.transition_matrix[c_prev] /= row_sum

    def reset(self):
        """Reset for new negotiation (keep learned parameters)."""
        self.belief = INITIAL_STATE_DIST.copy()
        self.debtor_history = []
        self.creditor_history = []

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'belief_state': self.get_belief_state(),
            'dominant_state': self.get_dominant_state(),
            'total_updates': self.total_updates,
            'debtor_history_len': len(self.debtor_history),
            'hmm_active': self.should_activate(),
        }
