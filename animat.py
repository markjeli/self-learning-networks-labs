import numpy as np
import torch
from torch import nn

import animat_fun as afun


class PolicyNetwork(nn.Module):
    def __init__(self, input_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.softmax(self.layer2(x), dim=-1)
        return x


# static map from lecture:
type_of_map = -1
obs_size = 3  # size of observable area e.g 3x3
if_cross = True  # observable area is cross-shaped e.g agent see only vertical and horizontal neighbouring cells

# other static map:
# type_of_map = -2
# obs_size = 3          # size of observable area e.g 3x3
# if_cross = False      # squared observable area

# # random map: middle:
# type_of_map = 0
# obs_size = 3
# if_cross = False

# # random map: middle:
# type_of_map = 2
# obs_size = 5
# if_cross = False

# # random map hard:
# type_of_map = 3
# obs_size = 7
# if_cross = False


def my_action(strategy, observation):
    # Convert observation to tensor and pass through policy network to get action probabilities
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    action_probs = strategy(observation_tensor).squeeze(0).detach().numpy()
    action = np.random.choice(
        len(action_probs), p=action_probs
    )  # Sample action based on probabilities
    return action


def animat_train(type_of_map, obs_size=3, if_cross=False):
    gamma = 0.97  # can be changed in training and test for the same value
    lr = 0.01
    # epsilon = ...
    number_of_episodes = 100

    # Initialize policy network and optimizer
    input_size = obs_size * obs_size if not if_cross else 2 * obs_size - 1
    policy_net = PolicyNetwork(input_size)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    for epi in range(number_of_episodes):
        map = afun.generate_map(
            type_of_map
        )  # can be generated for more than one episode
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4 * (num_of_rows + num_of_columns)

        if_end = False
        step_number = 0
        sum_of_discounted_rewards = 0
        cumulated_gamma = 1

        trajectories = []

        while not if_end:
            step_number += 1

            # square region observed by agent:
            observation = afun.observable_region(map, obs_size, position, if_cross)
            observation_flat = (
                observation[observation != -1] if if_cross else observation.flatten()
            )

            action = my_action(policy_net, observation_flat)

            new_position, reward = afun.transition_and_reward(map, position, action)

            trajectories.append((observation_flat, action, reward))

            if reward > 0 or step_number > max_num_of_steps:
                if_end = True

            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

        # Process trajectories for policy update
        returns = []
        G = 0
        for _, _, reward in reversed(trajectories):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (
            returns.std() + 1e-8
        )  # Normalize returns

        observations, actions, _ = zip(*trajectories)
        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)

        # Compute loss and update policy network
        action_probs = policy_net(observations)
        log_probs = torch.log(action_probs[range(len(actions)), actions])
        loss = -(log_probs * returns).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return policy_net


def animat_test(strategy, type_of_map, obs_size=3, if_cross=False):
    gamma = 0.97  # can be changed in training and test for the same value
    number_of_episodes = 100

    mean_sum_of_discounted_rewards = 0

    for epi in range(number_of_episodes):
        map = afun.generate_map(
            type_of_map
        )  # can be generate for more than one episode
        num_of_rows, num_of_columns = np.shape(map)
        position = afun.start_position(map)
        max_num_of_steps = 4 * (num_of_rows + num_of_columns)

        if_end = False
        step_number = 0
        sum_of_discounted_rewards = 0
        cumulated_gamma = 1
        path = []

        while if_end == False:  # episode steps loop
            step_number += 1

            # square region observed by agent:
            observation = afun.observable_region(map, obs_size, position, if_cross)
            observation_flat = (
                observation[observation != -1] if if_cross else observation.flatten()
            )
            # print(str(observation))
            action = my_action(strategy, observation_flat)

            new_position, reward = afun.transition_and_reward(map, position, action)

            path.append([*position, action, *new_position, reward])

            if (reward > 0) | (step_number > max_num_of_steps):
                if_end = True

            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

        mean_sum_of_discounted_rewards += sum_of_discounted_rewards / number_of_episodes

        print(
            "episode "
            + str(epi)
            + ": steps = "
            + str(step_number)
            + " sum_of_rewards = "
            + str(sum_of_discounted_rewards)
        )

        if epi < 5:
            afun.save_map_and_path(map, path, epi)
            afun.save_text_animation(map, path, epi)

    print("after " + str(number_of_episodes) + " episodes:")
    print("mean sum of discounted rewards = " + str(mean_sum_of_discounted_rewards))


strategy = animat_train(type_of_map, obs_size, if_cross)
animat_test(strategy, type_of_map, obs_size, if_cross)
