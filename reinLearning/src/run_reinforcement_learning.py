# src/train_q_learning.py
import numpy as np
from reinLearning.src.reinforcement_learning_env import initialize_q_matrix, choose_action, update_q_matrix

# Initialize states and rewards
R = np.array([[0, 1, -1],
              [1, 0, -1],
              [1, 1, 0]])  # Reward matrix

# Hyperparameters
gamma = 0.8  # Discount factor
learning_rate = 0.9
episodes = 10000  # Number of episodes

# Train the model
def train_q_learning():
    Q = initialize_q_matrix(R.shape[0], R.shape[1])

    for _ in range(episodes):
        state = np.random.randint(0, R.shape[0])  # Start at a random state
        while True:
            action = choose_action(state, Q, R)
            next_state = action
            Q = update_q_matrix(Q, R, state, action, next_state, learning_rate, gamma)
            if action == 2:  # End state
                break
            state = action

    return Q

if __name__ == "__main__":
    trained_q_matrix = train_q_learning()
    print("Trained Q-matrix:")
    print(trained_q_matrix)
