from logging import critical

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import animat_fun as afun


class ActorNetwork(nn.Module):
    def __init__(self, input_size):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.softmax(self.output(x))
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.output(x)
        return x


# static map from lecture:
# type_of_map = -1
# obs_size = 3  # size of observable area e.g 3x3
# if_cross = True  # observable area is cross-shaped e.g agent see only vertical and horizontal neighbouring cells

# other static map:
type_of_map = -2
obs_size = 3          # size of observable area e.g 3x3
if_cross = False      # squared observable area

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
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    action_probs = strategy(observation_tensor).squeeze(0).detach().numpy()
    action = np.random.choice(
        len(action_probs), p=action_probs
    )  # Sample action based on probabilities
    return action


def animat_train(type_of_map, obs_size=3, if_cross=False):
    gamma = 1  # can be changed in training and test for the same value
    lr = 0.001
    number_of_episodes = 200

    # Initialize policy network and optimizer
    input_size = obs_size**2
    actor_net = ActorNetwork(input_size)
    critic_net = CriticNetwork(input_size)
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic_net.parameters(), lr=lr)

    losses = {"actor": [], "critic": []}

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

        actor_epi_loss = []
        critic_epi_loss = []

        while not if_end:
            step_number += 1

            observation = afun.observable_region(
                map, obs_size, position, if_cross
            ).flatten()
            action = my_action(actor_net, observation)
            new_position, reward = afun.transition_and_reward(map, position, action)

            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            value = critic_net(observation_tensor)

            next_observation = afun.observable_region(
                map, obs_size, new_position, if_cross
            ).flatten()
            next_value = critic_net(torch.tensor(next_observation, dtype=torch.float32))

            td_error = reward + gamma * next_value - value

            action_probs = actor_net(observation_tensor)
            log_prob = torch.log(action_probs[action])
            actor_loss = -log_prob * td_error.detach() * cumulated_gamma

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # critic_loss = td_error.pow(2).mean()
            critic_loss = td_error.pow(2)
            # critic_loss = (next_value - value).pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            if reward > 0 or step_number > max_num_of_steps:
                if_end = True

            position = new_position
            sum_of_discounted_rewards += reward * cumulated_gamma
            cumulated_gamma *= gamma

            actor_epi_loss.append(actor_loss.item())
            critic_epi_loss.append(critic_loss.item())

        if epi % 100 == 0:
            print(f"Episode {epi}, Total Reward: {sum_of_discounted_rewards}")

        losses["actor"].append(sum(actor_epi_loss)/len(actor_epi_loss))
        losses["critic"].append(sum(critic_epi_loss)/len(critic_epi_loss))

    plt.plot(losses["actor"], label="Actor Loss")
    plt.plot(losses["critic"], label="Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()

    return actor_net


def animat_test(strategy, type_of_map, obs_size=3, if_cross=False):
    gamma = 0.98  # can be changed in training and test for the same value
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
            observation = afun.observable_region(
                map, obs_size, position, if_cross
            ).flatten()
            # print(str(observation))
            action = my_action(strategy, observation)

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
