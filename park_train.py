import numpy as np
import parking_model as pm
from tqdm import tqdm
from numba import njit

iht_size = 4096  # Rozmiar tablicy kodowania (tile coding)
num_tilings = 8  # Liczba pokryć
tile_size = [
    0.2,
    0.2,
    np.pi / 8,
    np.pi / 8,
    0.5,
]  # Rozmiar kafelka dla (x, y, kąt, kąt skrętu, prędkość)
offsets = [
    (i / num_tilings) * np.array(tile_size) for i in range(num_tilings)
]  # Przesunięcia dla każdego pokrycia


# # przykładowa nagroda za krok - nie wiem czy dobra
# def nagroda_za_krok(param_fiz, stan, czy_kolizja, czy_zatrzymanie):
#     wartosc = 0
#     x = stan[0]
#     y = stan[1]
#     alfa = stan[2]
#     odl_xy_kw = x * x + y * y
#     alfa_zred = 0
#     if param_fiz.if_side_parking_place:
#         if np.abs(alfa) > np.pi / 2:
#             alfa_zred = np.pi - np.abs(alfa)
#         else:
#             alfa_zred = np.abs(alfa)
#     else:
#         alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)
#
#     alfa_zred = alfa_zred / (odl_xy_kw + 0.5)
#
#     ocena_odl = 1 / (odl_xy_kw + 0.5) - 1
#     ocena_alfa = alfa_zred - 0.5
#
#     # jeśli V==0 nagroda na podstawie odległości
#
#     if czy_kolizja:
#         wartosc = -1
#     elif czy_zatrzymanie:
#         wartosc = min(ocena_odl, ocena_alfa)
#     else:
#         wartosc = 0
#
#     return wartosc


def nagroda_za_krok(param_fiz, stan, czy_kolizja, krok):
    x, y, alfa = stan
    odl_xy = np.sqrt(x**2 + y**2)

    # Normalize the angle to be within [0, pi] range
    alfa = np.abs(alfa % (2 * np.pi))
    if alfa > np.pi:
        alfa = 2 * np.pi - alfa
    if alfa > np.pi / 2:
        alfa = np.pi - alfa

    # Calculate the distance and angle penalties
    distance_penalty = odl_xy
    angle_penalty = alfa / np.pi

    # Calculate the step penalty if the number of steps exceeds the rational number
    rational_num_of_steps = (
        param_fiz.park_depth + param_fiz.street_width + param_fiz.street_length
    ) / (param_fiz.Vmod * param_fiz.dt)
    step_penalty = max(krok - rational_num_of_steps, 0) / rational_num_of_steps

    # Calculate the reward
    if czy_kolizja:
        reward = -1
    else:
        reward = -(distance_penalty + angle_penalty + step_penalty)

    return reward


def choose_action(param_fiz, stan, model):
    # tutaj należy wykorzystać wyuczoną strategię w czystej eksploatacji
    # strategia może być np. reprezentowana aproksymatorem funkcji użyteczności
    # ..........................................
    # ..........................................
    # Eksploatacja: wybór najlepszej akcji
    actions = [
        [angle, speed]
        for angle in np.arange(
            -param_fiz.wheel_turn_angle_max,
            param_fiz.wheel_turn_angle_max + np.pi / 8,
            np.pi / 8,
        )
        for speed in np.arange(-param_fiz.Vmod, param_fiz.Vmod + 0.5, 0.5)
    ]
    q_values = np.array([Q_value(stan, action, model) for action in actions])
    best_action = actions[np.argmax(q_values)]
    action = best_action

    kat, V = action
    # czy_zatrzymanie = False  # na razie (można przyjąć True np. gdy |V| < próg)
    czy_zatrzymanie = np.abs(V) < 0.1
    return kat, V, czy_zatrzymanie


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
            kat, V, czy_zatrzymanie = choose_action(param_fiz, stan, model)

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


def get_tiles(state, action, iht_size=4096):
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
    wheel_turn_angle_max = np.pi / 4
    Vmod = 2
    actions = [
        [angle, speed]
        for angle in np.arange(
            -wheel_turn_angle_max, wheel_turn_angle_max + np.pi / 8, np.pi / 8
        )
        for speed in np.arange(-Vmod, Vmod + 0.5, 0.5)
    ]
    if np.random.rand() < epsilon:
        # Eksploracja: losowy wybór akcji
        action = [
            np.random.uniform(-wheel_turn_angle_max, wheel_turn_angle_max),
            np.random.uniform(-Vmod, Vmod),
        ]  # Przykład zakresów
    else:
        # Eksploatacja: wybór najlepszej akcji
        q_values = np.array([Q_value(state, action, weights) for action in actions])
        best_action = actions[np.argmax(q_values)]
        action = best_action
    return action


def update_weights(
    weights, state, action, reward, next_state, next_action, gamma, alpha
):
    # Obliczamy wektor funkcji cech dla stanu i akcji oraz kolejnego stanu i akcji
    current_features = get_tiles(state, action)  # Binarna tablica dla (s, a)
    next_features = get_tiles(next_state, next_action)  # Binarna tablica dla (s', a')

    # Obliczamy wartość Q dla obecnego stanu i akcji oraz kolejnego stanu i akcji
    current_Q = np.dot(current_features, weights)
    next_Q = np.dot(next_features, weights)

    # Obliczamy błąd TD
    td_error = reward + gamma * next_Q - current_Q

    # Aktualizacja wag: wagi = wagi + α * błąd TD * cechy
    weights += alpha * td_error * current_features


def park_train():
    liczba_epizodow = 2000
    alpha = 0.001  # wsp.szybkosci uczenia(moze byc funkcja czasu)
    epsilon = 0.1  # wsp.eksploracji(moze byc funkcja czasu)
    gamma = 0.99  # Czynnik dyskontujący

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

    param_fiz = pm.GlobalVar()  # parametry fizyczne parkingu i pojazdu
    # Inicjalizujemy wagi
    weights = np.zeros(iht_size)

    for epizod in tqdm(range(liczba_epizodow)):
        # Wybieramy stan poczatkowy:
        nr_stanup = epizod % liczba_stanow_poczatkowych
        state = stany_poczatkowe[nr_stanup, :]

        krok = 0
        czy_kolizja = False
        czy_zatrzymanie = False
        while czy_zatrzymanie == False:
            krok = krok + 1

            action = epsilon_greedy_policy(state, weights, epsilon)
            angle, V = action

            # wyznaczenie nowego stanu:
            next_state, sr_obrotu, czy_kolizja = pm.model_of_car(
                param_fiz, state, angle, V
            )

            if czy_kolizja or (krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            # reward = nagroda_za_krok(
            #     param_fiz, next_state, czy_kolizja, czy_zatrzymanie
            # )
            reward = nagroda_za_krok(param_fiz, next_state, czy_kolizja, krok)
            next_action = epsilon_greedy_policy(next_state, weights, epsilon)
            update_weights(
                weights, state, action, reward, next_state, next_action, gamma, alpha
            )

            state = next_state

        # co jakis czas test z wygenerowaniem historii do pliku:
        if epizod % 1000 == 0:
            print("epizod %d\n" % epizod)
            park_test(param_fiz, stany_poczatkowe, weights, "historia_park.txt")

    # sprawdzenie czy system dobrze uogólnia dla dowolnych stanów początkowych:
    stany_pocz_losowe = pm.random_initial_states(pm.GlobalVar(), 20)
    print("Test dla losowych stanów początkowych:")
    park_test(param_fiz, stany_pocz_losowe, weights, "historia_park_los.txt")


ocena_koncowa_maks = pm.final_score(
    pm.GlobalVar(), [0, 0, -np.pi], if_collision=False, num_of_steps=100
)
print("najlepsza możliwa ocena końcowa = " + str(ocena_koncowa_maks))

park_train()
