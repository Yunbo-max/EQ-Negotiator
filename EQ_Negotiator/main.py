# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2025-04-14 00:47:39
# @Last Modified by:   Yunbo
# @Last Modified time: 2025-04-14 00:48:03


from EQ_Negotiator.system import EmotionInteractionSystem

def main():
    system = EmotionInteractionSystem()
    customer_emotions = ["Sadness", "Fear", "Anger", "Anger", "Sadness", "Disgust", "Surprise", "Joy"]
    
    current_robot_emotion = None
    for cust_emotion in customer_emotions:
        next_robot_emotion = system.get_next_emotion(cust_emotion, current_robot_emotion)
        strategy = "HMM" if system._should_use_hmm() else "Payoff"
        print(f"Customer: {cust_emotion.ljust(8)} -> Robot: {next_robot_emotion.ljust(8)} | Strategy: {strategy}")
        current_robot_emotion = next_robot_emotion

if __name__ == "__main__":
    main()