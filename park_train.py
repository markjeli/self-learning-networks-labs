import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from numba import njit
from tqdm import tqdm

import parking_model as pm
from model import PolicyNet

# Global Variables
GLOBAL_VARS = pm.GlobalVar()

## Training Hyperparameters
ALPHA = 0.001  # Współczynnik uczenia
EPS_START = 0.9
EPS_END = 0.05
GAMMA = 0.99  # Czynnik dyskontujący
LAMBDA = 0.8  # Parametr świeżości (śladu)

number_of_episodes = 5000

## Actions
PREDEFINED_ACTIONS = [
    [angle, speed]
    for angle in np.arange(
        -GLOBAL_VARS.wheel_turn_angle_max,
        GLOBAL_VARS.wheel_turn_angle_max + np.pi / 8,
        np.pi / 8,
    )
    for speed in np.arange(-GLOBAL_VARS.Vmod, GLOBAL_VARS.Vmod + 1, 1)
    if not speed == 0
]
# PREDEFINED_ACTIONS.append([0, 0])

## Tiles
IHT_SIZE = 4096  # Rozmiar tablicy kodowania (tile coding)
num_tilings = 8  # Liczba pokryć
tile_size = [
    0.5,
    0.5,
    np.pi / 8,
    np.pi / 8,
    0.5,
]  # Rozmiar kafelka dla (x, y, kąt, kąt skrętu, prędkość)
# offsets = [
#     (i / num_tilings) * np.array(tile_size) for i in range(num_tilings)
# ]  # Przesunięcia dla każdego pokrycia

offsets = [
    (i - num_tilings // 2) / num_tilings * np.array(tile_size)
    for i in range(num_tilings)
] # Przesunięcia dla każdego pokrycia (centrowanie)


## Batching
BATCH_SIZE = 10
experience_buffer = []

## Cache
cached_features = []

## Logging
episode_rewards = []
episode_steps = []

### Code for neural network training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUMBER_OF_FEATURES = 3 # 3 for state (x, y, angle)
NUMBER_OF_ACTIONS = len(PREDEFINED_ACTIONS)

def optimize_model():
    if len(experience_buffer) < BATCH_SIZE:
        return

    batch = random.sample(experience_buffer, BATCH_SIZE)

    state_batch = torch.cat([state for state, _, _, _ in batch])
    action_batch = torch.cat([action for _, action, _, _ in batch])
    reward_batch = torch.cat([reward for _, _, reward, _ in batch])
    next_state_batch = torch.cat([next_state for _, _, _, next_state in batch])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute Q-values for the next states using the policy_net
    with torch.inference_mode():
        next_state_values = policy_net(next_state_batch).max(1).values


    # Compute the TD target
    td_target = reward_batch + GAMMA * next_state_values

    # Compute the TD error
    td_error = td_target - state_action_values

    # # Compute gradients for the policy_net
    loss = td_error.pow(2).mean()  # Using MSE for simplicity
    loss.backward()  # Backpropagate to compute gradients


    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    # Update weights directly using the computed gradients
    for param in policy_net.parameters():
        param.data += ALPHA * td_error.mean().item() * param.grad.data  # TODO: Check if we add or subtract the gradient

    # Zero gradients after updating
    policy_net.zero_grad()


## Neural Model
policy_net = PolicyNet(NUMBER_OF_FEATURES, NUMBER_OF_ACTIONS).to(DEVICE)
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=ALPHA, amsgrad=True)
##############################






# przykładowa nagroda za krok - nie wiem czy dobra
def nagroda_za_krok(param_fiz, stan, czy_kolizja, czy_zatrzymanie):
    x, y, alfa = stan
    odl_xy_kw = np.sqrt(x**2 + y**2)
    if param_fiz.if_side_parking_place:
        if np.abs(alfa) > np.pi / 2:
            alfa_zred = np.pi - np.abs(alfa)
        else:
            alfa_zred = np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)

    alfa_zred = alfa_zred / (odl_xy_kw + 0.5)

    ocena_odl = 1 / (odl_xy_kw + 0.5) - 1
    ocena_alfa = alfa_zred - 0.5

    # jeśli V==0 nagroda na podstawie odległości

    if czy_kolizja:
        wartosc = -1
    elif czy_zatrzymanie:
        wartosc = min(ocena_odl, ocena_alfa)
    else:
        wartosc = 0

    return wartosc


def reward_function(state, last_state, if_collision, if_stopped):
    x, y, angle = state
    last_x, last_y, last_angle = last_state

    # Distance to goal (0, 0)
    distance_to_goal = np.sqrt(x**2 + y**2)
    last_distance_to_goal = np.sqrt(last_x**2 + last_y**2)

    # Angle error (aligning to goal orientation)
    angle_error = min(abs(angle), abs(abs(angle) - np.pi))

    # Reward for reducing distance
    progress_reward = max(0, last_distance_to_goal - distance_to_goal)

    # Angle alignment reward (smoothed with cosine)
    angle_alignment_reward = np.cos(angle_error)

    # Collision penalty
    collision_penalty = -50.0 if if_collision else 0

    # Stopping reward
    stopping_reward = 0
    goal_tolerance = 1
    angle_tolerance = np.radians(5)
    if (
        if_stopped
        and distance_to_goal < goal_tolerance
        and angle_error < angle_tolerance
    ):
        stopping_reward = 10.0

    # Combine rewards
    reward = (
        0.8 * progress_reward
        + 0.2 * angle_alignment_reward
        + stopping_reward
        + collision_penalty
    )
    return reward


def reward_function2(state, if_collision, if_stopped):
    distance_to_goal = np.sqrt(state[0] ** 2 + state[1] ** 2)
    angle_error = min(abs(state[2]), abs(abs(state[2]) - np.pi))

    distance_reward = -np.clip(distance_to_goal, 0, 1)
    angle_reward = np.cos(angle_error)

    if if_collision:
        return -1.0
    elif if_stopped:
        return distance_reward + angle_reward
    else:
        return 0.1 * distance_reward  # Gradual reward for progress

def final_score_reward(state, if_collision, num_of_steps):
    x, y, angle = state
    distance = np.sqrt(x * x + y * y)

    if GLOBAL_VARS.if_side_parking_place:
        if np.abs(angle) > np.pi / 2:
            angle_reduced = np.pi - np.abs(angle)
        else:
            angle_reduced = np.abs(angle)
    else:
        angle_reduced = np.abs(np.abs(angle) - np.pi / 2)

    rational_num_of_steps = (
        GLOBAL_VARS.park_depth + GLOBAL_VARS.street_width + GLOBAL_VARS.street_length
    ) / (GLOBAL_VARS.Vmod * GLOBAL_VARS.dt)
    excess_step_num = max(num_of_steps - rational_num_of_steps, 0)

    collision_penalty = -100 if if_collision else 0

    score = (
        1
        / (1 + distance)
        / (1 + angle_reduced * 2)
        / (1 + int(if_collision))
        / (1 + excess_step_num / rational_num_of_steps)
    )

    score += collision_penalty

    return score


def check_if_stopped(state) -> bool:
    x, y, angle = state
    distance_to_goal = np.sqrt(x**2 + y**2)
    angle_error = min(abs(angle), abs(abs(angle) - np.pi))

    goal_tolerance = 1
    angle_tolerance = np.radians(5)

    return distance_to_goal < goal_tolerance and angle_error < angle_tolerance


def choose_action(state, weights, param_fiz=GLOBAL_VARS):
    state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.inference_mode():
        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        action_idx = policy_net(state).max(1).indices.view(1, 1)

    action = PREDEFINED_ACTIONS[action_idx]
    angle, V = action
    if_stopped = check_if_stopped(state.squeeze()) or V == 0
    return angle, V, if_stopped


# test parkowania - nie wolno niczego zmieniać!
def park_test(param_fiz, stany_poczatkowe, model, nazwa_pliku):
    pm.park_save("param.txt", param_fiz)
    phist = open(nazwa_pliku, "w")
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape
    sr_ocena_koncowa = 0
    sr_liczba_krokow = 0
    for epizod in range(liczba_stanow_poczatkowych):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod % liczba_stanow_poczatkowych
        stan = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan zgodnie z wyuczoną strategią:
            kat, V, czy_zatrzymanie = choose_action(stan, model, param_fiz)

            # zapis kroku historii:
            # phist.write(str(epizod + 1) + "  " + str(krok) + "  " + str(stan[0]) + "  " + str(stan[1]) + "  " + str(stan[2]) + "  " + str(kat) + "  " + str(V) + "\n")
            phist.write(
                "%d %d %.4f %.4f %.4f %.4f %.4f\n"
                % ((epizod + 1), krok, stan[0], stan[1], stan[2], kat, V)
            )
            # wyznaczenie nowego stanu:
            nowystan, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, stan, kat, V)

            if (czy_kolizja) | (krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            stan = nowystan
        ocena_koncowa = pm.final_score(param_fiz, nowystan, czy_kolizja, krok)
        sr_ocena_koncowa += ocena_koncowa / liczba_stanow_poczatkowych
        sr_liczba_krokow = sr_liczba_krokow + krok / liczba_stanow_poczatkowych
        print(
            "w %d epizodzie ocena parkowania = %g, liczba krokow = %d"
            % (epizod, ocena_koncowa, krok)
        )

    print("srednia ocena końcowa na epizod = %g" % (sr_ocena_koncowa))
    print("srednia liczba krokow = %g" % (sr_liczba_krokow))
    phist.close()
    return sr_ocena_koncowa


# Funkcja do generowania unikalnych indeksów kafelków
@njit
def tile_hash(indices):
    return sum([index * (i + 1) for i, index in enumerate(indices)]) % IHT_SIZE


def get_tiles(state, action):
    tile_vector = np.zeros(IHT_SIZE)  # Binarna tablica o rozmiarze iht_size
    combined_state_action = np.concatenate((state, action))  # Łączymy stan i akcję

    for offset in offsets:
        # Obliczamy indeks kafelka dla danej przesuniętej kombinacji stan+akcja
        tile_index = np.floor((combined_state_action + offset) / tile_size).astype(int)
        # Hashujemy indeks kafelka i ustawiamy go na 1 w binarnej tablicy
        index = tile_hash(tile_index)
        tile_vector[index] = 1  # Aktywujemy kafelek w tablicy

    return tile_vector


def Q_value(state, action, weights):
    features = get_tiles(
        state, action
    )  # Otrzymujemy binarną tablicę aktywnych kafelków
    return np.dot(features, weights)  # Mnożenie macierzy, aby uzyskać wartość Q


# Wybór akcji z polityką epsilon-greedy
def epsilon_greedy_policy(state, weights, epsilon, step=0):
    if_stopped = False
    if np.random.rand() < epsilon:
        # Eksploracja: losowy wybór akcji
        action_idx = torch.tensor(np.random.choice(len(PREDEFINED_ACTIONS)), device=DEVICE).view(1, 1)
    else:
        # Eksploatacja: wybór najlepszej akcji
        with torch.inference_mode():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_idx = policy_net(state).max(1).indices.view(1, 1)

        # angle, V, if_stopped = choose_action(state, weights)
        # action = [angle, V]
    if step > 200:
        if_stopped = True
    return action_idx, if_stopped


def update_weights(
    weights, state, action, reward, next_state, z, gamma, alpha
):
    current_features = get_tiles(state, action)
    next_q_values = np.array(
        [Q_value(next_state, a, weights) for a in PREDEFINED_ACTIONS]
        # [np.dot(get_tiles(next_state, a), weights) for a in PREDEFINED_ACTIONS]
    )

    z = gamma * LAMBDA * z + current_features

    td_error = (
        reward + gamma * np.max(next_q_values) - np.dot(current_features, weights)
    )

    weights += alpha * td_error * z
    return weights, z


def update_weights_mini_batch(
    weights, state, action, reward, next_state, z, gamma, alpha
):
    experience_buffer.append((state, action, reward, next_state))
    if len(experience_buffer) >= BATCH_SIZE:
        batch = random.sample(experience_buffer, BATCH_SIZE)
        for state, action, reward, next_state in batch:
            weights, z = update_weights(weights, state, action, reward, next_state, z, gamma, alpha)
        experience_buffer.clear()
    return weights, z

def park_train():
    alpha = ALPHA
    epsilon = EPS_START
    gamma = GAMMA

    stany_poczatkowe_1 = np.array(
        [
            [9.1, 4.6, 0],
            [6.3, 5.06, 0],
            [9.6, 3.15, 0],
            [7.3, 5.75, 0],
            [10.1, 6.21, 0],
        ],
        dtype=float,
    )  # z prawej przodem w prawo
    stany_poczatkowe_2 = np.array(
        [
            [9.1, 4.6, np.pi],
            [6.3, 5.06, np.pi],
            [9.6, 3.15, np.pi],
            [7.3, 5.75, np.pi],
            [10.1, 6.21, np.pi],
        ],
        dtype=float,
    )  # z prawej przodem w lewo
    stany_poczatkowe_3 = np.array(
        [
            [-9.1, 4.6, 0],
            [-6.3, 5.06, 0],
            [-9.6, 3.15, 0],
            [-7.3, 5.75, 0],
            [-10.1, 6.21, 0],
        ],
        dtype=float,
    )  # z lewej przodem w prawo
    stany_poczatkowe_4 = np.array(
        [
            [-9.1, 4.6, np.pi],
            [-6.3, 5.06, np.pi],
            [-9.6, 3.15, np.pi],
            [-7.3, 5.75, np.pi],
            [-10.1, 6.21, np.pi],
        ],
        dtype=float,
    )  # z lewej przodem w lewo
    stany_poczatkowe = stany_poczatkowe_1
    liczba_stanow_poczatkowych, lparam = stany_poczatkowe.shape

    # Inicjalizujemy wagi
    weights = np.zeros(IHT_SIZE)
    z = np.zeros(IHT_SIZE)


    for episode in tqdm(range(number_of_episodes)):
        epsilon = max(EPS_END, epsilon * 0.99)  # stopniowe zmniejszanie eksploracji

        z[:] = 0

        # Wybieramy stan poczatkowy:
        nr_stanup = episode % liczba_stanow_poczatkowych
        state = stany_poczatkowe[nr_stanup, :]
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        step = 0
        if_collision = False
        if_stopped = False
        total_reward = 0
        while not if_stopped:
            step = step + 1

            action_idx, if_stopped = epsilon_greedy_policy(state, weights, epsilon, step)
            angle, V = PREDEFINED_ACTIONS[action_idx]

            # wyznaczenie nowego stanu:
            next_state, rotation_center, if_collision = pm.model_of_car(
                GLOBAL_VARS, state.squeeze(), angle, V
            )
            next_state = torch.tensor(next_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            if if_collision or (step >= GLOBAL_VARS.max_number_of_steps):
                if_stopped = True

            # reward = reward_function(next_state.squeeze(), state.squeeze(), if_collision, if_stopped)
            reward = reward_function2(next_state.squeeze(), if_collision, if_stopped)
            # reward = final_score_reward(next_state.squeeze(), if_collision, step)
            reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

            experience_buffer.append((state, action_idx, reward, next_state))
            optimize_model()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_steps.append(step)

        # co jakis czas test z wygenerowaniem historii do pliku:
        if episode % 100 == 0:
            print("epizod %d\n" % episode)
            park_test(GLOBAL_VARS, stany_poczatkowe, weights, "historia_park.txt")

    # sprawdzenie czy system dobrze uogólnia dla dowolnych stanów początkowych:
    stany_pocz_losowe = pm.random_initial_states(pm.GlobalVar(), 20)
    print("Test dla losowych stanów początkowych:")
    park_test(GLOBAL_VARS, stany_pocz_losowe, weights, "historia_park_los.txt")


ocena_koncowa_maks = pm.final_score(
    pm.GlobalVar(), [0, 0, -np.pi], if_collision=False, num_of_steps=100
)
print("najlepsza możliwa ocena końcowa = " + str(ocena_koncowa_maks))

park_train()

# After training, plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards, label="Rewards")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(episode_steps, label="Steps")
plt.xlabel("Episodes")
plt.ylabel("Steps")
plt.legend()

plt.show()
