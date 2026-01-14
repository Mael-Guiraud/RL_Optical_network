# -*- coding: utf-8 -*-

import numpy as np
import math
import copy
from functools import lru_cache

from greedy import min_avg_delay, mrd_buffer

 
def generate_constrained_binary_matrix(rows, cols, load=0.9):
    """
    Generate a binary matrix with a constrained number of ones per column.
    """
    # Calculate minimal and maximal number of ones allowed
    min_ones = int(load * cols)
    max_ones = cols
    num_ones = np.random.randint(min_ones, max_ones + 1)
    # Initialize a zero matrix
    matrix = np.zeros((rows, cols), dtype=int)

    # Generate random indices to place ones
    total_elements = rows * cols
    indices = np.random.choice(total_elements, num_ones, replace=False)

    for index in indices:
        row = index // cols
        col = index % cols
        matrix[row, col] = 1

    return matrix


 
def get_columns_with_ones(matrix):
    """
    Find the indices of columns that contain at least one '1' in the matrix.
    """
    # Find columns that contain at least one '1'
    columns_with_ones = np.any(matrix == 1, axis=0)
    # Get indices of these columns
    column_indices = np.where(columns_with_ones)[0]
    return column_indices


 
def create_actions_values(P, c):
    """
    Create a list of action values for players.
    """
    return [0] + [i for i in range(c, P)] + [i for i in range(P + 1, P + c)]


@lru_cache(maxsize=None)
def generate_arrangement_cached(N, P):
    """
    Recursively generate arrangements and cache results to improve performance.
    """
    if N == 0:
        return []
    if N == 1:
        return tuple([tuple([i]) for i in range(P)])  # Using tuples for cache
    else:
        result = []
        for i in range(P):
            sub_words = generate_arrangement_cached(N - 1, P)  # Recursive call with cache
            for j in sub_words:
                if i not in j:
                    result.append((i,) + j)  # Combine tuples
        return tuple(result)


 
def generate_arrangement(N, P):
    """
    Generate arrangements for N elements from P possibilities.
    """
    # Call the cached function then convert to lists
    cached_result = generate_arrangement_cached(N, P)
    return [list(x) for x in cached_result]  # Convert tuples to lists


 
def init_profile(matrix, P):
    """
    Initialize the profile for each player based on the binary matrix.
    """
    # Find the number of ones in each column
    ones_counts = np.sum(matrix == 1, axis=0)
    # Get indices of columns that contain at least one '1'
    columns_with_ones = np.where(ones_counts > 0)[0]
    profile = []
    for col in columns_with_ones:
        num_ones = ones_counts[col]
        nb_elems = int(math.factorial(P) / math.factorial(P - num_ones))
        probabilities = np.full(nb_elems, 1 / nb_elems)
        arrangements = generate_arrangement(num_ones, P)
        # Append to profile: [column index, number of ones, number of elements, probabilities, arrangements]
        profile.append([col, num_ones, nb_elems, probabilities, arrangements])
    return profile


 
def draw_random_strategy_one_player(player_index, profile):
    """
    Draw a random strategy for one player based on their profile.
    """
    probabilities = profile[player_index][3]
    return np.random.choice(len(probabilities), p=probabilities)


 
def get_chosen_action(player_index, profile, strategy_index, F, actions, P):
    """
    Get the chosen action for a player based on their strategy.
    """
    # Copy the column from matrix F
    column_F = F[:, profile[player_index][0]].copy()
    strategy = profile[player_index][4][strategy_index].copy()
    action_indices = np.where(column_F == 1)[0]
    for idx in action_indices:
        column_F[idx] = actions[strategy.pop(0)]
    return column_F


 
def add_column_to_matrix(matrix, new_column):
    """
    Add a new column to the existing matrix.
    """
    # Check that the number of rows in the new column matches the matrix
    if matrix.shape[0] != len(new_column):
        raise ValueError("The size of the new column does not match the number of rows of the matrix")
    # Add the new column to the matrix
    new_matrix = np.hstack((matrix, new_column.reshape(-1, 1)))
    return new_matrix


 
def compute_fs(S, F, P):
    """
    Compute the Fs matrix based on S and F matrices.
    """
    Fs = np.zeros((len(S), len(S[0])), dtype=int)
    for i in range(len(S)):
        for j in range(len(S[0])):
            if F[i][j] == 1:
                Fs[i][(j + S[i][j]) % P] += 1
    return Fs


 
def cols_j(S, F, j, P):
    """
    Compute the number of collisions for a given column j.
    """
    cols = 0
    for i in range(len(S)):
        if F[i][j] == 1:
            for iprime in range(len(S)):
                for k in range(len(S[0])):
                    if j != k and F[iprime][k] == 1 and (j + S[i][j]) % P == (k + S[iprime][k]) % P:
                        cols += 1
    return cols


 
def init_buffer_matrix(P):
    """
    Initialize the buffer matrix for the network.
    """
    buffer_matrix = np.zeros(P, dtype=int)
    return buffer_matrix


 
def update_buffer_matrix(buffer_matrix, period, begin, delay):
    """
    Update the buffer matrix based on the delay and period.
    """
    all_buff = False
    if delay >= period:
        all_buff = True
    first_value = begin
    last_value = (begin + delay) % period
    if all_buff:
        buffer_matrix += 1
    if first_value < last_value:
        buffer_matrix[first_value:last_value] += 1
    if first_value > last_value:
        buffer_matrix[first_value:] += 1
        buffer_matrix[:last_value] += 1

 
def fill_buffer_matrix(buffer_matrix, S, F, P):
    """
    Fill the buffer matrix for all players.
    """
    for i in range(len(S)):
        for j in range(len(S[0])):
            if F[i][j] == 1:
                update_buffer_matrix(buffer_matrix, P, j, S[i][j])


 
def buffer_matrix_all_players(S, F, P):
    """
    Create the buffer matrix for all players.
    """
    buffer_matrix = init_buffer_matrix(P)
    fill_buffer_matrix(buffer_matrix, S, F, P)
    return buffer_matrix


 
def buffer_matrix_one_player(S, F, P, player_column_index):
    """
    Create the buffer matrix for a single player.
    """
    buffer_matrix = init_buffer_matrix(P)
    for i in range(len(S)):
        if F[i][player_column_index] == 1:
            update_buffer_matrix(buffer_matrix, P, player_column_index, S[i][player_column_index])
    return buffer_matrix


 
def Ocap_j(S, F, P, player_column_index, B, bm_all, bm_player):
    """
    Compute the overcapacity for a single player.
    """
    if isinstance(bm_player, np.ndarray) and bm_player.size == 0:
        return 0
    elif isinstance(bm_player, list) and bm_player == []:
        return 0

    Ocap = 0
    for t in range(P):
        Ocap += max(0, bm_all[t] - B) * bm_player[t]
    return Ocap


 
def colls_S(S, F, P, player_columns):
    """
    Compute the total number of collisions in the system.
    """
    total_collisions = 0
    for col_idx in player_columns:
        total_collisions += cols_j(S, F, col_idx, P)
    return total_collisions


 
def Ocap_S(S, F, P, B, bm_all, bm_players, player_columns):
    """
    Compute the total overcapacity in the system.
    """
    total_Ocap = 0
    for idx, col_idx in enumerate(player_columns):
        total_Ocap += Ocap_j(S, F, P, col_idx, B, bm_all, bm_players[idx])
    return total_Ocap


 
def max_coll_j(S, F, P, player_columns):
    """
    Find the maximum number of collisions for any player.
    """
    max_collisions = 0
    for col_idx in player_columns:
        coll_j = cols_j(S, F, col_idx, P)
        max_collisions = max(max_collisions, coll_j)
    return max_collisions


 
def max_Ocap(S, F, P, B, bm_all, bm_players, player_columns):
    """
    Find the maximum overcapacity for any player.
    """
    max_ocap = 0
    for idx, col_idx in enumerate(player_columns):
        ocap_j_value = Ocap_j(S, F, P, col_idx, B, bm_all, bm_players[idx])
        max_ocap = max(max_ocap, ocap_j_value)
    return max_ocap


 
def max_ratio_delay(S, taumin):
    """
    Compute the maximum ratio of delay over minimal delay (taumin).
    """
    maxi = 0
    for i in range(len(S)):
        maxi = max(maxi, max(S[i]) / max(taumin[i], 1))
    return maxi


 
def max_ratio_delay_player(F, S, taumin, player_column_index, delay):
    """
    Compute the maximum ratio of delay over minimal delay (taumin) for a single player.
    """
    maxi = 0
    for i in range(len(S)):
        if F[i][player_column_index] == 1:
            max_delay_line = 0

            for j in range(len(S[i])):
                if (S[i][j] + delay[i][j]) / taumin[i] > max_delay_line:
                    max_delay_line = (S[i][j] + delay[i][j]) / taumin[i]
            maxi = max(maxi, max_delay_line)
    return maxi


 
def is_valid(total_collisions, total_Ocap):
    """
    Check if the current configuration is valid (no collisions and no overcapacity).
    """
    return total_collisions == 0 and total_Ocap == 0


 
def maxglob(P, mct, taumin, delay_max):
    """
    Compute the global maximum delay ratio.
    """
    return (delay_max + P + mct) / max(min(taumin), 1)


 
def maxglob_player(F, P, mct, taumin, player_column_index, delay_max_player):
    """
    Compute the global maximum delay ratio for a single player.
    """
    mini_taumin = max(taumin)
    for i in range(len(F)):
        if F[i][player_column_index] == 1:
            mini_taumin = min(mini_taumin, taumin[i])
    return (delay_max_player + P + mct) / mini_taumin


 
def Glob_s(S, F, P, B, taumin, mct, total_collisions, total_Ocap, delay_max):
    """
    Compute the global delay ratio.
    """
    if is_valid(total_collisions, total_Ocap):
        return max_ratio_delay(S, taumin)
    return maxglob(P, mct, taumin, delay_max)


 
def MRS_j(S, F, P, B, taumin, mct, player_column_index, delay, total_collisions, total_Ocap, delay_max):
    """
    Compute the maximum ratio of delay over minimal delay for a single player.
    """
    if is_valid(total_collisions, total_Ocap):
        return max_ratio_delay_player(F, S, taumin, player_column_index, delay)
    return maxglob(P, mct, taumin, delay_max)


 
def MRS_j_2(S, F, P, B, taumin, mct, player_column_index, delay, total_collisions, total_Ocap, delay_max_player):
    """
    Alternative computation of the maximum ratio of delay over minimal delay for a single player.
    """
    if is_valid(total_collisions, total_Ocap):
        return max_ratio_delay_player(F, S, taumin, player_column_index, delay)
    return maxglob_player(F, P, mct, taumin, player_column_index, delay_max_player)


 
def init_taumin(N, P, mode):
    """
    Initialize the minimal delays (taumin) for each route.
    """
    if mode == "P":
        return np.random.randint(1, P, N)
    elif mode == "2P":
        return np.random.randint(P, 2 * P, N)
    return np.random.randint(2 * P, 3 * P, N)


 
def player_cost_3(F, P, taumin, mct, player_column_index, alphac, alphao, glob_S_value, coll_player, ocap_player, max_colls, max_ocap, mrs_player_2, delay_max_player):
    """
    Compute the cost for a single player.
    """
    a = mrs_player_2 / maxglob_player(F, P, mct, taumin, player_column_index, delay_max_player)
    return 0.5 * a + alphac * (coll_player / max(1, max_colls)) + alphao * (ocap_player / max(1, max_ocap))


 
def compute_utility_opti(pc, cmax, cmin):
    """
    Compute the utility of a player based on their cost.
    """
    return (cmax - pc) / max(1, cmax - cmin)


 
def alphas(previous_colls, previous_ocap, total_collisions, total_Ocap):
    """
    Adjust the alpha values based on previous collisions and overcapacities.
    """
    if not previous_colls:
        previous_colls.append(total_collisions)
    else:
        previous_colls[0] = max(previous_colls[0], total_collisions)

    if not previous_ocap:
        previous_ocap.append(total_Ocap)
    else:
        previous_ocap[0] = max(previous_ocap[0], total_Ocap)

    a = total_collisions / max(1, previous_colls[0])
    b = total_Ocap / max(1, previous_ocap[0])
    c = max(a - b, 0)
    d = max(b - a, 0)
    if a >= b:
        return (min(0.25 + c / 4, 0.4), max(0.25 - c / 4, 0.1))
    return (max(0.25 - d / 4, 0.1), min(0.25 + d / 4, 0.4))


 
def max_delay(delay):
    """
    Return the maximum delay from the delay matrix.
    """
    return np.max(delay)


 
def max_delay_player(F, delay, player_column_index):
    """
    Return the maximum delay for a specific player.
    """
    delays = [delay[i][player_column_index] for i in range(len(delay)) if F[i][player_column_index] == 1]
    return max(delays) if delays else 0


 
def avg_delay(S, delay):
    """
    Compute the average delay in the system.
    """
    total_delay = 0
    count = 0
    for i in range(len(delay)):
        for j in range(len(delay[i])):
            if S[i][j] != 0 or delay[i][j] != 0:
                total_delay += S[i][j] + delay[i][j]
                count += 1
    return total_delay / count if count != 0 else 0

def initialize_probabilities_with_greedy(profile, greedy_strategies, greedy_prob=0.7):
    for j in range(len(profile)):
        num_strategies = profile[j][2]
        if num_strategies == 1:
            probabilities = np.array([1.0])
        else:
            probabilities = np.full(num_strategies, (1 - greedy_prob) / (num_strategies - 1))
            greedy_strategy_index = greedy_strategies[j]
            probabilities[greedy_strategy_index] = greedy_prob
        profile[j][3] = probabilities

def get_greedy_strategies_indices(profile, greedy_solution, F, actions, P):
    """
    Trouve l'indice de la stratégie correspondant à la solution gloutonne pour chaque joueur.

    Paramètres:
        profile (list): Liste des profils des joueurs.
        greedy_solution (ndarray): Matrice S de la solution gloutonne.
        F (ndarray): Matrice binaire représentant les liens.
        actions (list): Liste des valeurs d'actions possibles.
        P (int): Période du système.

    Retourne:
        List[int]: Liste des indices des stratégies gloutonnes pour chaque joueur.
    """
    greedy_strategies = []
    for j in range(len(profile)):
        player_column_index = profile[j][0]
        possible_strategies = profile[j][4]
        num_strategies = profile[j][2]
        greedy_action = greedy_solution[:, player_column_index]
        # Parcourir toutes les stratégies possibles pour ce joueur
        found = False
        for idx in range(num_strategies):
            action = get_chosen_action(j, profile, idx, F, actions, P)
            action = action.astype(int)
            if np.array_equal(action, greedy_action):
                greedy_strategies.append(idx)
                found = True
                break
        if not found:
            # Si aucune stratégie ne correspond, vous pouvez choisir de mettre une probabilité uniforme
            # ou ajouter la stratégie gloutonne aux stratégies possibles
            greedy_strategies.append(0)  # Par défaut, on choisit l'indice 0
            print(f"Aucune stratégie correspondante trouvée pour le joueur {j}.")
    return greedy_strategies

 
def run_LRI(P, n, F, taumin, delay, mct, B, b, nb_step):
    """
    Run the Learning Rate Iteration (LRI) algorithm.
    """
    # Initialize actions and profile
    actions = create_actions_values(P, mct)
    profile = init_profile(F, P)
    nb_players = len(profile)
    player_columns = [profile[i][0] for i in range(nb_players)]


    arrivals = []
    for j in range(P):
        arrivals.append([])
        for i in range(len(F)):
            if F[i][j] == 1:
                arrivals[j].append([(0, taumin[i], delay[i][j]), [i, j]])
    res = min_avg_delay(copy.deepcopy(arrivals), mct, B, mrd_buffer)
    S_current = np.zeros((n, P), dtype=int)
    for elem in res[2]:
        i, j = elem[1]
        S_current[i][j] = elem[0]
    greedy_solution = S_current.copy()

    greedy_strategies = get_greedy_strategies_indices(profile, greedy_solution, F, actions, P)
 
    greedy_prob = 0.1
    initialize_probabilities_with_greedy(profile, greedy_strategies, greedy_prob)
    # Initialize variables
    delay_max = max_delay(delay)
    delay_max_players = [max_delay_player(F, delay, profile[i][0]) for i in range(nb_players)]
    previous_costs = [[[] for _ in range(profile[j][2])] for j in range(nb_players)]
    previous_colls = []
    previous_ocap = []
    picked_strategies = [0 for _ in range(nb_players)]
    val_cols = []
    val_ocap = []
    val_glob = []
    val_proba = []
    val_utility_max = []
    val_utility_min = []
    val_utility_avg = []
    val_avg_dl = []
    best_val = math.inf
    best_S = None

    for t in range(nb_step):
        S = np.zeros((n, 0), dtype=int)
        id_in_profile = 0
        for col in range(len(F[0])):
            # If the column contains only zeros, add it to S
            if np.sum(F[:, col]) == 0:
                S = add_column_to_matrix(S, F[:, col])
            elif col in player_columns:
                picked_strategies[id_in_profile] = draw_random_strategy_one_player(id_in_profile, profile)
                new_col = get_chosen_action(id_in_profile, profile, picked_strategies[id_in_profile], F, actions, P)
                new_col = new_col.reshape(-1, 1)
                S = add_column_to_matrix(S, new_col)
                id_in_profile += 1
            else:
                S = add_column_to_matrix(S, F[:, col])
        S = S.astype(int)

        # Compute buffer matrices
        bm_all = buffer_matrix_all_players(S, F, P)
        bm_players = [buffer_matrix_one_player(S, F, P, profile[j][0]) for j in range(nb_players)]

        # Compute collisions and overcapacities

        # Compute per-player metrics
        colls_players = []
        ocap_players = []
        mrs_players_2 = []
        for j in range(nb_players):
            player_column_index = profile[j][0]
            coll_player = cols_j(S, F, player_column_index, P)
            ocap_player = Ocap_j(S, F, P, player_column_index, B, bm_all, bm_players[j])
            colls_players.append(coll_player)
            ocap_players.append(ocap_player)
        total_collisions = sum(colls_players)
        total_Ocap = sum(ocap_players)
        for j in range(nb_players):
            player_column_index = profile[j][0]
            mrs_player_2 = MRS_j_2(S, F, P, B, taumin, mct, player_column_index, delay, total_collisions, total_Ocap, delay_max_players[j])
            mrs_players_2.append(mrs_player_2)

        # Compute maximum collisions and overcapacities
        max_colls = max(colls_players)
        max_ocap = max(ocap_players)
        glob_S_value = max(mrs_players_2)
        alphac, alphao = alphas(previous_colls, previous_ocap, total_collisions, total_Ocap)


        # Initialize utility variables
        avg_utility = 0
        max_utility = 0
        min_utility = 1
        avg_proba = 0

        for j in range(nb_players):
            player_column_index = profile[j][0]
            Pc = player_cost_3(
                F, P, taumin, mct, player_column_index, alphac, alphao, glob_S_value,
                colls_players[j], ocap_players[j], max_colls, max_ocap, mrs_players_2[j], delay_max_players[j]
            )

            if t == 0:
                # Initialize previous costs
                for i in range(profile[j][2]):
                    previous_costs[j][i].append(Pc - 1)
                    previous_costs[j][i].append(Pc)

            previous_costs[j][picked_strategies[j]][0] = min(previous_costs[j][picked_strategies[j]][0], Pc)
            previous_costs[j][picked_strategies[j]][1] = max(previous_costs[j][picked_strategies[j]][1], Pc)
            utility = compute_utility_opti(
                Pc,
                previous_costs[j][picked_strategies[j]][1],
                previous_costs[j][picked_strategies[j]][0]
            )
            avg_utility += utility
            max_utility = max(max_utility, utility)
            min_utility = min(min_utility, utility)

            # Update probabilities
            probabilities = profile[j][3]
            probabilities[picked_strategies[j]] += b * utility * (1 - probabilities[picked_strategies[j]])
            for i in range(profile[j][2]):
                if i != picked_strategies[j]:
                    probabilities[i] -= b * utility * probabilities[i]
            avg_proba += max(probabilities) - min(probabilities)

        val_cols.append(total_collisions)
        val_ocap.append(total_Ocap)
        val_glob.append(glob_S_value)
        val_utility_avg.append(avg_utility / nb_players)
        val_utility_max.append(max_utility)
        val_utility_min.append(min_utility)
        val_proba.append(avg_proba / nb_players)

        if glob_S_value < best_val:
            best_val = glob_S_value
            best_S = S.copy()

        val_avg_dl.append(avg_delay(S, delay))

    # Further processing can be added here as needed
    arrivals =[]
    for j in range(P):
        arrivals.append([])
        for i in range(len(F)):
            if(F[i][j]==1):
                arrivals[j].append([(0,taumin[i],delay[i,j]),[i,j]])
                
    res= min_avg_delay(copy.deepcopy(arrivals),mct,B,mrd_buffer)
    max_delay_routes= [ 0 for _ in range(n)]

    for elem in res[2]:
        max_delay_routes[elem[1][0]] = max(max_delay_routes[elem[1][0]],(elem[0]+delay[elem[1][0]][elem[1][1]])/taumin[elem[1][0]])

    greedy_value = max(max_delay_routes)
    minglob = []
    minim = val_glob[0]
    for i in val_glob:
        minim=min(minim,i)
        minglob.append(minim)

    return val_cols, val_ocap, val_glob, val_utility_avg, val_utility_max, val_utility_min, val_proba, val_avg_dl,greedy_value,minglob,res,profile, best_S