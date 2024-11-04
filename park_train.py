import numpy as np

import parking_model as pm
from tqdm import tqdm
from numba import njit

# Global Variables
GLOBAL_VARS = pm.GlobalVar()

## Training Hyperparameters
ALPHA = 0.002
EPSILON = 1.0
gamma = 0.90  # Czynnik dyskontujący

number_of_episodes = 5000

## Tiles
iht_size = 8192  # Rozmiar tablicy kodowania (tile coding)
num_tilings = 8  # Liczba pokryć
tile_size = [
    0.1,
    0.1,
    np.pi / 8,
    np.pi / 8,
    0.5,
]  # Rozmiar kafelka dla (x, y, kąt, kąt skrętu, prędkość)
offsets = [
    (i / num_tilings) * np.array(tile_size) for i in range(num_tilings)
]  # Przesunięcia dla każdego pokrycia

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
PREDEFINED_ACTIONS.append([0, 0])

## Batching
BATCH_SIZE = 10
experience_buffer = []


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


def nagroda_za_krok2(state, last_state, num_of_steps, if_collision, if_stopped):
    goal_tolerance = 0.2
    angle_tolerance = np.radians(5)
    x, y, angle = state
    last_x, last_y = last_state[:2]

    # Calculate distances
    distance = np.sqrt(x**2 + y**2)
    last_distance = np.sqrt(last_x**2 + last_y**2)

    # Reward for moving closer to the parking slot
    progress_reward = 1 if distance < last_distance else -1

    # Large reward for achieving the final parking position and orientation
    final_parked_reward = (
        100 if distance < goal_tolerance and abs(angle) < angle_tolerance else 0
    )

    # Penalty for excessive steps
    rational_num_of_steps = (
        GLOBAL_VARS.park_depth + GLOBAL_VARS.street_width + GLOBAL_VARS.street_length
    ) / (GLOBAL_VARS.Vmod * GLOBAL_VARS.dt)
    excess_step_penalty = -5 if num_of_steps > rational_num_of_steps else 0

    # Penalty for minimal progress
    position_change = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
    orientation_change = abs(angle - last_state[2])
    minimal_change_penalty = -1 if position_change < 0.01 and orientation_change < np.radians(1) else 0

    # Penalty if car stops prematurely
    stopped_penalty = -10 if if_stopped and distance > goal_tolerance else 0

    # Reward for correct angle orientation
    angle_reduced = (
        abs(angle) if GLOBAL_VARS.if_side_parking_place else abs(abs(angle) - np.pi / 2)
    )
    angle_reward = 2 / (1 + angle_reduced * 5)

    # Collision penalty
    collision_penalty = -50 if if_collision else 0

    # Exploration bonus for new movements
    exploration_reward = 0.1 if position_change > 0.01 or orientation_change > np.radians(1) else 0

    # Combine all components
    total_reward = (
        progress_reward
        + final_parked_reward
        + excess_step_penalty
        + minimal_change_penalty
        + stopped_penalty
        + angle_reward
        + collision_penalty
        + exploration_reward
    )

    return total_reward

def nagroda_za_krok3(state, last_state, num_of_steps, if_collision, if_stopped, last_speed):
    goal_tolerance = 0.2
    angle_tolerance = np.radians(5)
    x, y, angle = state
    last_x, last_y = last_state[:2]

    # Calculate distances
    distance = np.sqrt(x**2 + y**2)
    last_distance = np.sqrt(last_x**2 + last_y**2)

    # Reward for moving closer to the parking slot
    progress_reward = 1 if distance < last_distance else -1

    # Large reward for achieving the final parking position and orientation
    final_parked_reward = 200 if distance < goal_tolerance and abs(angle) < angle_tolerance else 0

    # Penalty for excessive steps
    rational_num_of_steps = (
        GLOBAL_VARS.park_depth + GLOBAL_VARS.street_width + GLOBAL_VARS.street_length
    ) / (GLOBAL_VARS.Vmod * GLOBAL_VARS.dt)
    excess_step_penalty = -5 if num_of_steps > rational_num_of_steps else 0

    # Penalty if car stops prematurely
    stopped_penalty = -10 if if_stopped and distance > goal_tolerance else 0

    # Reward for correct angle orientation
    angle_reduced = (
        abs(angle) if GLOBAL_VARS.if_side_parking_place else abs(abs(angle) - np.pi / 2)
    )
    angle_reward = 2 / (1 + angle_reduced * 5)

    # Collision penalty
    collision_penalty = -50 if if_collision else 0

    # Oscillation penalty for frequent speed changes
    oscillation_penalty = -5 if np.sign(state[1]) != np.sign(last_speed) else 0

    # Penalty for selecting the [0, 0] action in early training steps
    zero_action_penalty = -10 if (state == [0, 0]) and (distance > goal_tolerance) else 0

    # Combine all components
    total_reward = (
        progress_reward
        + final_parked_reward
        + excess_step_penalty
        + stopped_penalty
        + angle_reward
        + collision_penalty
        + oscillation_penalty
        + zero_action_penalty
    )

    return total_reward


def reward_function(state, collision, step, czy_zatrzymanie):
    distance_threshold = 0.1  # Maksymalna odległość uznawana za zaparkowanie
    collision_penalty = -100
    excess_step_penalty = -0.05  # Kara za każdy krok powyżej max_steps
    x, y, angle = state
    distance = np.sqrt(x**2 + y**2)
    distance_reward = max(0, 1 - 5 * distance / distance_threshold)

    angle_tolerance = np.pi / 18
    orientation_error = min(
        abs(angle), abs(np.pi - abs(angle))
    )  # Bierzemy pod uwagę kąt względem 0 lub 180
    orientation_reward = 1 if orientation_error < angle_tolerance else 0

    if collision:
        return collision_penalty

    # Kara za nadmierną liczbę kroków
    max_steps = 100
    step_penalty = -0.01 + (excess_step_penalty * max(0, step - max_steps))

    total_reward = distance_reward + orientation_reward + step_penalty
    return total_reward


def choose_action(state, weights, param_fiz=GLOBAL_VARS):
    q_values = np.array(
        [Q_value(state, action, weights) for action in PREDEFINED_ACTIONS]
    )
    best_action = PREDEFINED_ACTIONS[np.argmax(q_values)]
    action = best_action

    angle, V = action
    if_stopped = V == 0
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
def tile_hash(indices, iht_size):
    return sum([index * (i + 1) for i, index in enumerate(indices)]) % iht_size


def get_tiles(state, action, iht_size=iht_size):
    tile_vector = np.zeros(iht_size)  # Binarna tablica o rozmiarze iht_size
    combined_state_action = np.concatenate(
        (state, action), axis=0
    )  # Łączymy stan i akcję

    for offset in offsets:
        # Obliczamy indeks kafelka dla danej przesuniętej kombinacji stan+akcja
        tile_index = np.floor((combined_state_action + offset) / tile_size).astype(int)
        # Hashujemy indeks kafelka i ustawiamy go na 1 w binarnej tablicy
        index = tile_hash(tile_index, iht_size)
        tile_vector[index] = 1  # Aktywujemy kafelek w tablicy

    return tile_vector


def Q_value(state, action, weights):
    features = get_tiles(
        state, action
    )  # Otrzymujemy binarną tablicę aktywnych kafelków
    return np.dot(features, weights)  # Mnożenie macierzy, aby uzyskać wartość Q


# Wybór akcji z polityką epsilon-greedy
def epsilon_greedy_policy(state, weights, epsilon):
    if_stopped = False
    if np.random.rand() < epsilon:
        # Eksploracja: losowy wybór akcji
        action = [
            np.random.uniform(
                -GLOBAL_VARS.wheel_turn_angle_max, GLOBAL_VARS.wheel_turn_angle_max
            ),
            np.random.uniform(-GLOBAL_VARS.Vmod, GLOBAL_VARS.Vmod),
        ]  # Przykład zakresów
    else:
        # Eksploatacja: wybór najlepszej akcji
        angle, V, if_stopped = choose_action(state, weights)
        action = [angle, V]
    return action, if_stopped


def update_weights(
    weights, state, action, reward, next_state, next_action, gamma, alpha
):
    # Obliczamy wektor funkcji cech dla stanu i akcji oraz kolejnego stanu i akcji
    current_features = get_tiles(state, action)  # Binarna tablica dla (S, A)
    best_next_action = choose_action(next_state, weights)
    best_next_action = [best_next_action[0], best_next_action[1]]
    best_next_features = get_tiles(next_state, best_next_action)  # Binarna tablica dla (S', argmax_a Q(S', a))

    # Obliczamy wartość Q dla obecnego stanu i akcji oraz kolejnego stanu i akcji
    current_Q = np.dot(current_features, weights)
    best_next_Q = np.dot(best_next_features, weights)

    # Obliczamy błąd TD: r + γ * Q(S', argmax_a Q(S', a)) - Q(S, A)
    td_error = reward + gamma * best_next_Q - current_Q

    # Aktualizacja wag: wagi = wagi + α * błąd TD * cechy
    weights += alpha * td_error * current_features


def update_weights_mini_batch(
    weights, state, action, reward, next_state, next_action, gamma, alpha
):
    experience_buffer.append(
        (weights, state, action, reward, next_state, next_action, gamma, alpha)
    )
    if len(experience_buffer) >= BATCH_SIZE:
        for experience in experience_buffer:
            update_weights(*experience)
        experience_buffer.clear()


def park_train():
    alpha = ALPHA
    epsilon = EPSILON

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
    weights = np.zeros(iht_size)

    for episode in tqdm(range(number_of_episodes)):
        epsilon = max(0.1, epsilon * 0.99)  # stopniowe zmniejszanie eksploracji

        # Wybieramy stan poczatkowy:
        nr_stanup = episode % liczba_stanow_poczatkowych
        state = stany_poczatkowe[nr_stanup, :]

        step = 0
        if_collision = False
        if_stopped = False
        while if_stopped == False:
            step = step + 1

            action, if_stopped = epsilon_greedy_policy(state, weights, epsilon)
            angle, V = action

            # wyznaczenie nowego stanu:
            next_state, rotation_center, if_collision = pm.model_of_car(
                GLOBAL_VARS, state, angle, V
            )

            if if_collision or (step >= GLOBAL_VARS.max_number_of_steps):
                if_stopped = True

            # reward = reward_function(next_state, if_collision, step, if_stopped)
            reward = nagroda_za_krok(GLOBAL_VARS, next_state, if_collision, if_stopped)
            # reward = nagroda_za_krok2(next_state, state, step, if_collision, if_stopped)
            # reward = nagroda_za_krok3(next_state, state, step, if_collision, if_stopped, V)
            next_action, _ = epsilon_greedy_policy(next_state, weights, epsilon)
            # update_weights(
            #     weights, state, action, reward, next_state, next_action, gamma, alpha
            # )
            update_weights_mini_batch(
                weights, state, action, reward, next_state, next_action, gamma, alpha
            )

            state = next_state

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
