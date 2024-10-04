import numpy as np

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
sf.draw(reward_map, strategy, "random_strategy mean reward = " + str(random_strategy_mean_reward))

# miejsce na algorytm uczenia - modelem jest tablica Q
# (symulację epizodu można wziąć z funkcji sailor_test())
# ............................
# Uzupełnić skrypt sailor_train tak, aby pozwalał znaleźć optymalną strategię
# żeglarza w oparciu o tablicę użyteczności Q par (stan, akcja) i metodę symulacji
# Monte Carlo dla zadanych przez prowadzącego plansz (map).
# Należy porównać iterację strategii i iterację wartości pod kątem
# złożoności obliczeniowej i stabilności.

# Generate all possible state-action pairs
actions = [1, 2, 3, 4]
state_action_pairs = []
for x in range(num_of_rows):
    for y in range(num_of_columns):
        for a in actions:
            state_action_pairs.append(((x, y), a))

# Monte Carlo Iteration Strategy
while True:
    strategy_temp = strategy.copy()
    for state_action_pair in state_action_pairs:
        for episode in range(number_of_episodes):
            state = np.array(state_action_pair[0])
            action = state_action_pair[1]
            the_end = True if (state[1] >= num_of_columns - 1) else False
            nr_pos = 0
            while the_end == False:
                nr_pos = nr_pos + 1  # move number

                state_next, reward = sf.environment(state, action, reward_map)
                # Action choosing (1 - right, 2 - up, 3 - left, 4 - bottom):
                state = state_next  # going to the next state
                action = strategy[state[0], state[1]]

                # end of episode if maximum number of steps is reached or last column is reached
                if (nr_pos == num_of_steps_max) | (state[1] >= num_of_columns - 1):
                    the_end = True

                sum_of_rewards[episode] += gamma * reward

        Q[
            state_action_pair[0][0], state_action_pair[0][1], state_action_pair[1] - 1
        ] = np.mean(sum_of_rewards)
        sum_of_rewards = np.zeros([number_of_episodes], dtype=float)

    for x in range(num_of_rows):
        for y in range(num_of_columns):
            if y < num_of_columns - 1:
                strategy[x, y] = np.argmax(Q[x, y]) + 1
            else:
                strategy[x, y] = 0

    if np.array_equal(strategy, strategy_temp):
        break

sf.sailor_test(reward_map, strategy, 1000)
sf.draw(reward_map, strategy, "best_strategy")
