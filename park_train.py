import numpy as np
import parking_model as pm
from tqdm import tqdm

iht_size = 4096  # Rozmiar tablicy kodowania (tile coding)
num_tilings = 8  # Liczba pokryć
tile_size = [
    0.1,
    0.1,
    np.pi / 8,
    0.05,
    0.1,
]  # Rozmiar kafelka dla (x, y, kąt, kąt skrętu, prędkość)
offsets = [
    (i / num_tilings) * np.array(tile_size) for i in range(num_tilings)
]  # Przesunięcia dla każdego pokrycia


# przykładowa nagroda za krok - nie wiem czy dobra
def nagroda_za_krok(param_fiz, stan, czy_kolizja, czy_zatrzymanie):
    # tutaj należy ustalić nagrodę za każdy krok, tak by uczenie podążało
    # we właściwym kierunku:
    # ........................................
    # ........................................

    # przykładowe obliczenie nagrody za krok - nie wiem czy dobre:
    wartosc = 0
    x = stan[0]
    y = stan[1]
    alfa = stan[2]
    odl_xy_kw = x * x + y * y
    alfa_zred = 0
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


def choose_action(param_fiz, stan, model):
    # tutaj należy wykorzystać wyuczoną strategię w czystej eksploatacji
    # strategia może być np. reprezentowana aproksymatorem funkcji użyteczności
    # ..........................................
    # ..........................................
    # Eksploatacja: wybór najlepszej akcji
    best_action = None
    best_value = -np.inf
    for _ in range(100):  # Przeszukujemy przestrzeń akcji
        action = [
            np.random.uniform(-param_fiz.wheel_turn_angle_max, param_fiz.wheel_turn_angle_max),
            np.random.uniform(-param_fiz.Vmod, param_fiz.Vmod),
        ]
        value = Q_value(stan, action, model)
        if value > best_value:
            best_value = value
            best_action = action
    action = best_action

    # kat = -np.pi / 8  # jakiś kąt skrętu kół (na razie)
    # V = -param_fiz.Vmod  # jakaś prędkość (na razie)
    kat = action[0]
    V = action[1]
    czy_zatrzymanie = False  # na razie (można przyjąć True np. gdy |V| < próg)
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
def tile_hash(indices, iht_size):
    return sum([index * (i + 1) for i, index in enumerate(indices)]) % iht_size


# Funkcja do kodowania (stan + akcja) przy pomocy niestandardowego tile coding
def get_tiles(state, action, iht_size=4096):
    tile_indices = []
    combined_state_action = np.concatenate((state, action), axis=0)  # Łączymy stan i akcję

    for offset in offsets:
        # Przesunięcie w każdym pokryciu, podział przez tile_size, zaokrąglenie w dół
        tile_index = np.floor((combined_state_action + offset) / tile_size).astype(int)
        # Hashowanie indeksu kafelka, aby uzyskać unikalny integer
        tile_indices.append(tile_hash(tile_index, iht_size))

    return tile_indices


# Funkcja Q
def Q_value(state, action, weights):
    # Obliczamy wartość Q jako sumę aktywnych wag
    features = get_tiles(state, action)
    return sum(weights[f] for f in features)


# Wybór akcji z polityką epsilon-greedy
def epsilon_greedy_policy(state, weights, epsilon):
    wheel_turn_angle_max = np.pi / 4
    Vmod = 2
    if np.random.rand() < epsilon:
        # Eksploracja: losowy wybór akcji
        action = [
            np.random.uniform(-wheel_turn_angle_max, wheel_turn_angle_max),
            np.random.uniform(-Vmod, Vmod),
        ]  # Przykład zakresów
    else:
        # Eksploatacja: wybór najlepszej akcji
        best_action = None
        best_value = -np.inf
        for _ in range(100):  # Przeszukujemy przestrzeń akcji
            action = [
                np.random.uniform(-wheel_turn_angle_max, wheel_turn_angle_max),
                np.random.uniform(-Vmod, Vmod),
            ]
            value = Q_value(state, action, weights)
            if value > best_value:
                best_value = value
                best_action = action
        action = best_action
    return action


def park_train():
    liczba_epizodow = 200
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

    # inicjacja kodowania, wyznaczenie liczby parametrów (wag):
    # ........................................................
    # ........................................................

    # Parametry kodowania i aproksymacji


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

            # Wyznaczamy akcje a (kąt + kier. ruchu) w stanie stan z uwzględnieniem
            # eksploracji (np. metoda epsylon-zachlanna lub softmax lub jeszcze inna)
            # ........................................................
            # ........................................................
            action = epsilon_greedy_policy(state, weights, epsilon)
            angle = action[0]
            V = action[1]

            # wyznaczenie nowego stanu:
            next_state, sr_obrotu, czy_kolizja = pm.model_of_car(param_fiz, state, angle, V)

            if (czy_kolizja) | (krok >= param_fiz.max_number_of_steps):
                czy_zatrzymanie = True

            reward = nagroda_za_krok(param_fiz, next_state, czy_kolizja, czy_zatrzymanie)

            # Aktualizujemy wartosci Q dla aktualnego stanu i wybranej akcji:
            # ........................................................
            # ........................................................
            # w = w + ...
            next_action = epsilon_greedy_policy(next_state, weights, epsilon)
            features = get_tiles(state, action)
            next_Q = Q_value(next_state, next_action, weights)
            current_Q = Q_value(state, action, weights)
            td_error = reward + gamma * next_Q - current_Q
            for f in features:
                weights[f] += alpha * td_error

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
