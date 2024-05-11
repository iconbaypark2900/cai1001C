# src/q_learning.py
import numpy as np
import random

def initialize_q_matrix(num_states, num_actions):
    return np.zeros((num_states, num_actions))

def choose_action(state, q_matrix, reward_matrix):
    valid_actions = np.where(reward_matrix[state, :] >= 0)[0]
    return random.choice(valid_actions)

def update_q_matrix(q_matrix, reward_matrix, state, action, next_state, learning_rate, gamma):
    future_rewards = np.max(q_matrix[next_state, :])
    q_matrix[state, action] += learning_rate * (reward_matrix[state, action] + gamma * future_rewards - q_matrix[state, action])

    return q_matrix
