import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 10000  # number of training episodes (multi-stage processes)
gamma = 1.0  # discount factor

file_name = 'map_simple.txt'
#file_name = 'map_easy.txt'
#file_name = 'map_big.txt'
#file_name = 'map_spiral.txt'

reward_map = sf.load_data(file_name)
num_of_rows, num_of_columns = reward_map.shape

num_of_steps_max = int(
    5 * (num_of_rows + num_of_columns)
)  # maximum number of steps in an episode
Q = np.zeros(
    [num_of_rows, num_of_columns, 4], dtype=float
)  # trained action-value table of <state,action> pairs
sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

strategy = np.random.randint(
    low=1, high=5, size=np.shape(reward_map)
)  # random strategy
random_strategy_mean_reward = np.mean(sf.sailor_test(reward_map, strategy, 1000))
sf.draw_strategy(
    reward_map,
    strategy,
    "random_strategy mean reward = " + str(random_strategy_mean_reward),
)


for episode in range(number_of_episodes):
    alpha = sf.get_alpha(episode, number_of_episodes, num_of_rows * num_of_columns)
    epsilon = sf.get_epsilon(episode, number_of_episodes, num_of_rows * num_of_columns)
    state = np.zeros([2], dtype=int)  # initial state here [1 1]
    state[0] = np.random.randint(0, num_of_rows)
    the_end = True if (state[1] >= num_of_columns - 1) else False
    nr_pos = 0
    while the_end == False:
        nr_pos += 1
        action = sf.epsilon_greedy(state, Q, epsilon)
        state_next, reward = sf.environment(state, action, reward_map)
        best_next_action = np.argmax(Q[state_next[0], state_next[1], :]) + 1
        Q[state[0], state[1], action - 1] += alpha * (
            reward
            + gamma * Q[state_next[0], state_next[1], best_next_action - 1]
            - Q[state[0], state[1], action - 1]
        )
        state = state_next
        if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
            the_end = True

strategy = sf.strategy(Q)

sf.sailor_test(reward_map, strategy, 1000)
sf.draw_strategy(reward_map, strategy, "best_strategy")
