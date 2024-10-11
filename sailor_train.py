import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import sailor_funct as sf

number_of_episodes = 1000  # number of training epizodes (multi-stage processes)
gamma = 1.0  # discount factor


file_name = "map_small.txt"
# file_name = 'map_easy.txt'
# file_name = 'map_big.txt'
# file_name = 'map_spiral.txt'

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
sf.draw(
    reward_map,
    strategy,
    "random_strategy mean reward = " + str(random_strategy_mean_reward),
)


# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................


sf.sailor_test(reward_map, strategy, 1000)
sf.draw(reward_map, strategy, "best_strategy")
