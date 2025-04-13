# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-04-14 00:46:54
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-04-14 00:47:03
from .constants import EMOTIONS

PAYOFF_MATRIX = {
    'Joy': {'Joy': (4,4), 'Sadness': (2,3), 'Anger': (1,2), 'Fear': (2,1), 
            'Surprise': (3,3), 'Disgust': (2,2), 'Neutral': (3,3)},
    'Sadness': {'Joy': (3,2), 'Sadness': (3,3), 'Anger': (1,2), 'Fear': (2,1),
                'Surprise': (2,2), 'Disgust': (1,1), 'Neutral': (2,3)},
    'Anger': {'Joy': (2,1), 'Sadness': (2,1), 'Anger': (1,1), 'Fear': (1,0),
              'Surprise': (1,2), 'Disgust': (0,1), 'Neutral': (1,2)},
    'Fear': {'Joy': (1,2), 'Sadness': (1,2), 'Anger': (0,1), 'Fear': (2,2),
             'Surprise': (1,2), 'Disgust': (0,1), 'Neutral': (2,3)},
    'Surprise': {'Joy': (3,3), 'Sadness': (2,2), 'Anger': (2,1), 'Fear': (2,1),
                 'Surprise': (4,4), 'Disgust': (1,2), 'Neutral': (3,3)},
    'Disgust': {'Joy': (2,2), 'Sadness': (1,1), 'Anger': (1,0), 'Fear': (1,0),
                'Surprise': (2,1), 'Disgust': (2,2), 'Neutral': (2,2)},
    'Neutral': {'Joy': (3,3), 'Sadness': (2,3), 'Anger': (2,1), 'Fear': (3,2),
                'Surprise': (3,3), 'Disgust': (2,2), 'Neutral': (3,3)}
}

def get_optimal_response(customer_emotion):
    payoffs = PAYOFF_MATRIX[customer_emotion]
    max_payoff = max(payoffs[e][1] for e in EMOTIONS)
    best_emotions = [e for e in EMOTIONS if payoffs[e][1] == max_payoff]
    return random.choice(best_emotions)