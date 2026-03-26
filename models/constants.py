"""
EQ-Negotiator Constants
Emotion definitions, transition/emission/payoff matrices from the paper.
Based on psychological foundations (Thornton & Tamir 2017, Sun et al. 2023)
"""

import numpy as np

# ============================================================================
# EMOTION DEFINITIONS (Section 3.1)
# ============================================================================

EMOTIONS = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Neutral']
N_EMOTIONS = len(EMOTIONS)

EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {i: e for i, e in enumerate(EMOTIONS)}

# Negative emotions that trigger HMM activation (Eq. 4)
NEGATIVE_EMOTIONS = {'Sadness', 'Anger', 'Fear', 'Disgust'}

# ============================================================================
# HMM HIDDEN STATES (Section 3.2)
# ============================================================================

HIDDEN_STATES = ['Cooperative', 'Confrontational', 'Distressed', 'Strategic']
N_STATES = len(HIDDEN_STATES)

STATE_TO_IDX = {s: i for i, s in enumerate(HIDDEN_STATES)}

# ============================================================================
# TABLE 4(a): Transition Probabilities (Agent Emotion -> Next Agent Emotion)
# ============================================================================

TRANSITION_MATRIX = np.array([
    # To: Joy   Sad   Ang   Fear  Sur   Dis   Neu
    [0.50, 0.10, 0.05, 0.05, 0.20, 0.05, 0.05],  # From Joy
    [0.20, 0.40, 0.10, 0.10, 0.05, 0.10, 0.05],  # From Sadness
    [0.10, 0.20, 0.40, 0.10, 0.05, 0.10, 0.05],  # From Anger
    [0.10, 0.20, 0.10, 0.40, 0.05, 0.10, 0.05],  # From Fear
    [0.30, 0.05, 0.05, 0.05, 0.50, 0.05, 0.05],  # From Surprise
    [0.10, 0.20, 0.10, 0.10, 0.05, 0.40, 0.05],  # From Disgust
    [0.20, 0.10, 0.05, 0.05, 0.20, 0.05, 0.35],  # From Neutral
])

# ============================================================================
# TABLE 4(b): Emission Probabilities (Agent Emotion -> Client Emotion)
# ============================================================================

EMISSION_MATRIX = np.array([
    # Client: Joy  Sad   Ang   Fear  Sur   Dis   Neu
    [0.60, 0.05, 0.05, 0.05, 0.10, 0.05, 0.10],  # Agent Joy
    [0.05, 0.50, 0.20, 0.10, 0.05, 0.05, 0.05],  # Agent Sadness
    [0.05, 0.20, 0.50, 0.10, 0.05, 0.05, 0.05],  # Agent Anger
    [0.05, 0.20, 0.10, 0.50, 0.05, 0.05, 0.05],  # Agent Fear
    [0.10, 0.05, 0.05, 0.05, 0.60, 0.05, 0.10],  # Agent Surprise
    [0.05, 0.10, 0.20, 0.10, 0.05, 0.50, 0.05],  # Agent Disgust
    [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.40],  # Agent Neutral
])

# ============================================================================
# TABLE 5: Payoff Matrix (client_payoff, agent_payoff)
# ============================================================================

# Agent payoffs only (second element of each pair)
PAYOFF_MATRIX_AGENT = np.array([
    # Agent: Joy  Sad   Ang   Fear  Sur   Dis   Neu
    [4, 3, 2, 1, 3, 2, 3],  # Client Joy
    [2, 3, 2, 1, 2, 1, 3],  # Client Sadness
    [1, 1, 1, 0, 2, 1, 2],  # Client Anger
    [2, 2, 1, 2, 2, 1, 3],  # Client Fear
    [3, 2, 1, 1, 4, 2, 3],  # Client Surprise
    [2, 1, 0, 0, 1, 2, 2],  # Client Disgust
    [3, 3, 1, 2, 3, 2, 3],  # Client Neutral
])

# Client payoffs (first element of each pair)
PAYOFF_MATRIX_CLIENT = np.array([
    [4, 2, 1, 2, 3, 2, 3],  # Client Joy
    [3, 3, 1, 2, 2, 1, 2],  # Client Sadness
    [2, 2, 1, 1, 1, 0, 1],  # Client Anger
    [1, 1, 0, 2, 1, 0, 2],  # Client Fear
    [3, 2, 2, 2, 4, 1, 3],  # Client Surprise
    [2, 1, 1, 1, 2, 2, 2],  # Client Disgust
    [3, 2, 2, 3, 3, 2, 3],  # Client Neutral
])

# ============================================================================
# HMM STATE TRANSITION PRIORS
# ============================================================================

# Initial hidden state distribution
INITIAL_STATE_DIST = np.array([0.4, 0.2, 0.2, 0.2])

# State transition matrix P(S_{t+1} | S_t)
STATE_TRANSITION = np.array([
    # To: Coop   Conf   Dist   Strat
    [0.50, 0.15, 0.15, 0.20],  # From Cooperative
    [0.20, 0.40, 0.20, 0.20],  # From Confrontational
    [0.25, 0.15, 0.40, 0.20],  # From Distressed
    [0.20, 0.20, 0.15, 0.45],  # From Strategic
])

# State-to-emotion mapping: preferred emotions per hidden state
STATE_EMOTION_WEIGHTS = {
    'Cooperative': {'Joy': 0.35, 'Surprise': 0.25, 'Neutral': 0.25,
                    'Sadness': 0.05, 'Anger': 0.02, 'Fear': 0.03, 'Disgust': 0.05},
    'Confrontational': {'Anger': 0.15, 'Neutral': 0.30, 'Surprise': 0.15,
                        'Joy': 0.10, 'Sadness': 0.10, 'Fear': 0.10, 'Disgust': 0.10},
    'Distressed': {'Sadness': 0.30, 'Fear': 0.20, 'Neutral': 0.20,
                   'Joy': 0.10, 'Surprise': 0.05, 'Anger': 0.05, 'Disgust': 0.10},
    'Strategic': {'Neutral': 0.30, 'Joy': 0.20, 'Surprise': 0.20,
                  'Sadness': 0.10, 'Anger': 0.05, 'Fear': 0.05, 'Disgust': 0.10},
}

# ============================================================================
# EMOTION PROMPTS (Appendix 8.5, Figure 8)
# ============================================================================

EMOTION_PROMPTS = {
    "Joy": "Use an optimistic and positive tone, expressing confidence in finding a mutually beneficial solution. Show genuine warmth and encouragement.",
    "Sadness": "Use an empathetic and understanding tone, acknowledging the difficulty of the debtor's situation. Show genuine concern for their wellbeing.",
    "Anger": "Use a firm and assertive tone, emphasizing the urgency and seriousness of resolving this matter. Set clear boundaries without being hostile.",
    "Fear": "Use a cautious and concerned tone, highlighting potential consequences while seeking cooperative resolution.",
    "Surprise": "Use an engaging and unexpected approach, introducing creative solutions or new perspectives to break deadlocks.",
    "Disgust": "Use a disappointed but professional tone, expressing concern about the current trajectory while remaining constructive.",
    "Neutral": "Use a balanced and professional tone, focusing on facts, practical solutions, and objective assessment.",
}

# ============================================================================
# EXPERIMENT DEFAULTS
# ============================================================================

# HMM activation threshold (Section 4)
HMM_ACTIVATION_K = 4       # k negative emotions
HMM_ACTIVATION_N = 5       # within n turns
PAYOFF_THRESHOLD = 2.0      # tau_payoff for WSLS lose condition
MAX_DIALOG_TURNS = 30       # T_max
