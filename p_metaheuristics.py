
import copy
import math
import random

from p_supervisor import run_network_with_route_supervisor, packets_run_network, generate_uniform_supervisor_matrix

# ----------- RECUIT ------------

def neighbor_route_supervisor_matrix(matrix, noise=0.1, values_per_row=1, rows=1):
    """
    Perturbs a given matrix by modifying a specified number of values in a specified number of rows.

    Args:
       matrix (list of list of float): The original matrix to be perturbed.
       noise (float): The amount by which to perturb the selected values.
       values_per_row (int): The number of values to perturb in each selected row.
       rows (int): The number of rows to perturb in the matrix.

    Returns:
       list of list of float: A new matrix with the specified perturbations applied.
    """
    # Create a copy of the original matrix to avoid modifying it directly
    neighbor_matrix = [row[:] for row in matrix]

    # Select random rows to perturb
    rows_to_perturb = random.sample(range(len(neighbor_matrix)), min(rows, len(neighbor_matrix)))

    for row_index in rows_to_perturb:
        # Select random values to perturb in the chosen row
        values_to_perturb = random.sample(range(len(neighbor_matrix[row_index])), min(values_per_row, len(neighbor_matrix[row_index])))

        for value_index in values_to_perturb:
            neighbor_matrix[row_index][value_index] += random.choice([-1, 1]) * noise

        # Normalize the perturbed row
        row_sum = sum(abs(value) for value in neighbor_matrix[row_index])
        if row_sum != 0:
            neighbor_matrix[row_index] = [abs(value) / row_sum for value in neighbor_matrix[row_index]]

    return neighbor_matrix

def recuit_simule(instance, Fs, weight_matrix, nb_routes, period, c, B, initial_supervisor, temperature_threshold=0.1, initial_temperature=1.0, cooling_rate=0.95, noise=0.1, values_per_row=1, rows=1, solver_mode="POLYOPTI", mode="normal"):
    # Deep copy the initial supervisor matrix as our starting solution.
    current_supervisor = copy.deepcopy(initial_supervisor)

    if current_supervisor is None:
        # print("ERREUR : Current supervisor non valide")
        current_supervisor = generate_uniform_supervisor_matrix(instance, nb_routes)

    # Evaluate the initial solution.
    try:
        current_obj, current_error = run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, current_supervisor)
        #print("GOOD : Solution initiale valide : " + current_obj.__str__())
    except:
        print("ERREUR : Solution initiale non valide")
        return None, None, None, None

    best_supervisor = copy.deepcopy(current_supervisor)
    best_obj = current_obj
    best_error = current_error
    T = initial_temperature

    solutions=[]

    if mode == "normal":
        i = 0
        while T > temperature_threshold and best_obj != 0:
            i+=1
            # Generate a neighbor by perturbing one route's value.
            neighbor_supervisor = neighbor_route_supervisor_matrix(current_supervisor, noise, values_per_row, rows)
            neighbor_obj, current_error = run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, neighbor_supervisor, solver_mode)
            delta = neighbor_obj - current_obj

            # Accept the neighbor if it's better or with a probability if it's worse.
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_supervisor = neighbor_supervisor
                current_obj = neighbor_obj
                solutions.append(current_obj)
                # Update best solution if improved.
                if neighbor_obj < best_obj:
                    best_supervisor = neighbor_supervisor
                    best_obj = neighbor_obj
                    best_error = current_error

            #print(f"Iteration {i}: Current Obj = {current_obj}, Best Obj = {best_obj}, Temperature = {T}")
            # Cool down the temperature.
            T *= cooling_rate

    elif mode == "palier":
        while T > temperature_threshold and best_obj != 0:
            if T > 0.8:
                #print(f"Palier [1; 0.8[ - {T}")
                steps = 10
            elif T > 0.2:
                #print(f"Palier [0.8 ; 0.2[ - {T}")
                steps = 100
            else:
                #print(f"Palier [0.2 ; 0.01]  - {T}")
                steps = 10

            for j in range(steps) : #and best_obj != 0:
                # Generate a neighbor by perturbing one route's value.
                neighbor_supervisor = neighbor_route_supervisor_matrix(current_supervisor, noise, values_per_row, rows)
                neighbor_obj, current_error = run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, neighbor_supervisor, solver_mode)
                delta = neighbor_obj - current_obj

                # Accept the neighbor if it's better or with a probability if it's worse.
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_supervisor = neighbor_supervisor
                    current_obj = neighbor_obj
                    solutions.append(current_obj)
                    # Update best solution if improved.
                    if neighbor_obj < best_obj:
                        best_supervisor = neighbor_supervisor
                        best_obj = neighbor_obj
                        best_error = current_error

                #print(f"Iteration {_}: Current Obj = {current_obj}, Best Obj = {best_obj}")

            # Cool down the temperature.
            T *= cooling_rate
    else:
        print("ERREUR: Mode inconnu")
        return None, None, None

    return best_supervisor, best_obj, T, best_error, solutions

def neighbor_packet_supervisor_matrix(margin_matrix, noise=0.1, n_routes=1, n_packets=1, n_levels=1):
    """
    Perturbs the margin_matrix for packets by modifying a specified number of packet margin vectors.

    Args:
        margin_matrix (dict): Dict[route_id][packet_id][level], as from packets_matrix_builder()['margin_matrix']
        noise (float): The amount by which to perturb the selected margin value.
        n_routes (int): The number of routes to perturb.
        n_packets (int): The number of packets per route to perturb.
        n_levels (int): The number of levels (per packet) to perturb.

    Returns:
        dict: A new margin_matrix dict with the specified perturbations applied.
    """
    if margin_matrix is None:
        raise KeyError("Le dictionnaire ne contient pas la clé 'margin_matrix'.")

    neighbor = copy.deepcopy(margin_matrix)
    route_ids = list(neighbor.keys())
    routes_to_perturb = random.sample(route_ids, min(n_routes, len(route_ids)))

    for route_id in routes_to_perturb:
        packets = neighbor[route_id]
        if not packets:
            print("No packets")
            continue
        packet_indices = list(range(len(packets)))
        packets_to_perturb = random.sample(packet_indices, min(n_packets, len(packet_indices)))
        for packet_id in packets_to_perturb:
            margin_vec = packets[packet_id]
            if not margin_vec:
                print("No margin vector")
                continue
            level_indices = list(range(len(margin_vec)))
            levels_to_perturb = random.sample(level_indices, min(n_levels, len(level_indices)))
            for level in levels_to_perturb:
                margin_vec[level] = max(0.0, margin_vec[level] + random.choice([-1, 1]) * noise)
            # Normalize margin vector (sum to 1, all >= 0)
            total = sum(margin_vec)
            if total > 0:
                packets[packet_id] = [v / total for v in margin_vec]
            else:
                # If all zeros, fallback to uniform
                packets[packet_id] = [1.0 / len(margin_vec)] * len(margin_vec)

    return neighbor

def recuit_simule_packet(instance, Fs, weight_matrix, nb_routes, period, c, B, initial_supervisor, temperature_threshold=0.1, initial_temperature=1.0, cooling_rate=0.95, noise=0.1, n_routes=1, n_packets=1, n_levels=1, solver_mode="POLYOPTI", mode="normal"):
    """
    Simulated annealing-based optimization function to find the best packet supervisor matrix.
    This function attempts to minimize an objective function related to packet routing in
    a network by iteratively updating the supervisor matrix. It uses the simulated annealing
    heuristic, which involves probabilistically accepting suboptimal solutions to escape
    local minima, alongside a temperature-based mechanism to control the degree of
    exploration. Two modes of operation are supported: "normal" for straightforward simulated
    annealing and "palier" for a stepped temperature approach.

    Args:
        instance: The network topology or instance object representing the network's structure.
        Fs: The flow-specific parameter related to the network configuration.
        weight_matrix: Matrix representing weights or costs associated with different links or nodes in the network.
        nb_routes: Integer defining the number of routing paths evaluated in the network during execution.
        period: Integer parameter representing a time period or scheduling cycle for the network.
        c: A general-purpose parameter used in the optimization process, likely representing speed or capacity constraints.
        B: Bandwidth or another network constraint parameter affecting the optimization results.
        initial_supervisor: Initial matrix configuration for the supervisor values, used as the starting solution.
        temperature_threshold: Float defining the minimum threshold of temperature for the annealing process.
        initial_temperature: Float defining the temperature at the start of the annealing process.
        cooling_rate: Float to determine the rate at which the temperature decreases in each iteration.
        noise: Float affecting the randomness or perturbation level during the generation of neighboring solutions.
        n_routes: Integer representing the number of routes perturbed during the neighbor generation.
        n_packets: Integer representing the number of packet perturbations during the neighbor generation.
        n_levels: Integer representing the number of level perturbations during the neighbor generation.
        solver_mode: String determining the optimization solver type, default is "POLYOPTI".
        mode: String controlling the optimization mode, options are "normal" and "palier".

    Returns:
        tuple: A tuple containing the following:
            - best_supervisor: The best supervisor matrix obtained during the optimization process.
            - best_obj: The objective function value corresponding to the best supervisor matrix.
            - T: Final temperature value at the end of the annealing process.
            - best_error: Error or penalty value associated with the best solution obtained.

    Raises:
        None
    """
    # Deep copy the initial supervisor matrix as our starting solution.

    current_matrix = copy.deepcopy(initial_supervisor)

    # Evaluate the initial solution.
    try:
        current_obj, current_error = packets_run_network(instance, Fs, weight_matrix, nb_routes, period, c, B, current_matrix)
    except:
        print("ERREUR : Solution initiale non valide")
        return None, None, None

    best_supervisor = copy.deepcopy(current_matrix)
    best_obj = current_obj
    T = initial_temperature
    best_error = current_error
    solutions=[]

    if mode == "normal":
        i = 0
        while T > temperature_threshold and best_obj != 0:
            i+=1
            # Generate a neighbor by perturbing one route's value.
            neighbor_matrix = neighbor_packet_supervisor_matrix(current_matrix, noise, n_routes, n_packets, n_levels)
            # print(f"NS : {neighbor_matrix}")
            neighbor_obj, current_error = packets_run_network(instance, Fs, weight_matrix, nb_routes, period, c, B, neighbor_matrix)
            delta = neighbor_obj - current_obj

            # Accept the neighbor if it's better or with a probability if it's worse.
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_matrix = neighbor_matrix
                current_obj = neighbor_obj
                solutions.append(current_obj)
                # Update best solution if improved.
                if neighbor_obj < best_obj:
                    best_supervisor = neighbor_matrix
                    best_obj = neighbor_obj
                    best_error = current_error

            # print(f"Iteration {i}: Current Obj = {current_obj}, Best Obj = {best_obj}, Temperature = {T}")
            # Cool down the temperature.
            T *= cooling_rate

    elif mode == "palier":
        while T > temperature_threshold and best_obj != 0:
            if T > 8:
                #print(f"Palier [1; 0.8[ - {T}")
                steps = 10
            elif T > 2:
                #print(f"Palier [0.8 ; 0.2[ - {T}")
                steps = 100
            else:
                #print(f"Palier [0.2 ; 0.01]  - {T}")
                steps = 10

            for _ in range(steps) : # and best_obj != 0:
                # Generate a neighbor by perturbing one route's value.
                neighbor_matrix = neighbor_packet_supervisor_matrix(current_matrix, noise, n_routes, n_packets, n_levels)
                neighbor_obj, current_error = packets_run_network(instance, Fs, weight_matrix, nb_routes, period, c, B, neighbor_matrix)
                delta = neighbor_obj - current_obj

                # Accept the neighbor if it's better or with a probability if it's worse.
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_matrix = neighbor_matrix
                    current_obj = neighbor_obj
                    solutions.append(current_obj)
                    # Update best solution if improved.
                    if neighbor_obj < best_obj:
                        best_supervisor = neighbor_matrix
                        best_obj = neighbor_obj
                        best_error = current_error

                #print(f"Iteration {_}: Current Obj = {current_obj}, Best Obj = {best_obj}")

            # Cool down the temperature.
            T *= cooling_rate
    else:
        print("ERREUR: Mode inconnu")
        return None, None, None

    return best_supervisor, best_obj, T, best_error, solutions

# ----------- GENETIC ALGORITHM ------------

def normalize_row(row):
    """
    Normalise une ligne pour que la somme soit égale à 1 et que toutes les valeurs soient positives.
    """
    row = [abs(x) for x in row]  # Valeurs positives
    total = sum(row)
    if total == 0:
        return [1.0/len(row)] * len(row)  # Distribution uniforme si somme nulle
    return [x/total for x in row]

def generate_initial_population(size, supervisor_matrix):
    """
    Génère une population initiale avec des matrices normalisées.
    """
    population = []
    for _ in range(size):
        individual = copy.deepcopy(supervisor_matrix)
        for i, row in enumerate(individual):
            # Perturber et normaliser chaque ligne
            perturbed_row = [max(0, x + random.uniform(-0.1, 0.1)) for x in row]
            individual[i] = normalize_row(perturbed_row)
        population.append(individual)
    return population

def mutate(individual, noise=0.1):
    """
    Applique une mutation en préservant la normalisation.
    """
    mutated = copy.deepcopy(individual)
    for i, row in enumerate(mutated):
        if random.random() < 0.1:  # Probabilité de mutation par ligne
            perturbed_row = [max(0, x + random.uniform(-noise, noise)) for x in row]
            mutated[i] = normalize_row(perturbed_row)
    return mutated

def crossover(parent1, parent2):
    """
    Croisement avec normalisation des lignes résultantes.
    """
    child = []
    for row1, row2 in zip(parent1, parent2):
        # Croisement uniforme par ligne
        child_row = [random.choice([val1, val2]) for val1, val2 in zip(row1, row2)]
        child.append(normalize_row(child_row))
    return child

def fitness_function(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix):
    """
    Évalue la qualité d'une matrice superviseur.

    Args:
        instance, Fs, weight_matrix, nb_routes, period, c, B: Paramètres du problème.
        supervisor_matrix (list of list of float): Matrice superviseur à évaluer.

    Returns:
        float: Valeur de fitness (plus faible est meilleur).
    """
    try:
        return run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix)
    except:
        return float('inf')  # Pénalité pour les solutions invalides

def genetic_algorithm(instance, Fs, weight_matrix, nb_routes, period, c, B, initial_supervisor, population_size=20, generations=100, mutation_rate=0.1):
    """
    Implémente un algorithme génétique pour optimiser la matrice superviseur.

    Args:
        instance, Fs, weight_matrix, nb_routes, period, c, B: Paramètres du problème.
        initial_supervisor: Initial supervisor matrix
        population_size (int): Taille de la population.
        generations (int): Nombre de générations.
        mutation_rate (float): Taux de mutation.

    Returns:
        tuple: Meilleure matrice superviseur et sa valeur de fitness.
    """
    # Génération de la population initiale
    population = generate_initial_population(population_size, initial_supervisor)

    for generation in range(generations):
        # Évaluation de la fitness
        fitness_scores = [
            (individual, fitness_function(instance, Fs, weight_matrix, nb_routes, period, c, B, individual))
            for individual in population
        ]
        fitness_scores.sort(key=lambda x: x[1])  # Trier par fitness (ascendant)

        # Sélection des meilleurs individus
        selected = [individual for individual, _ in fitness_scores[:population_size // 2]]

        # Croisement et mutation pour générer la nouvelle population
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)

        population = new_population

        # Affichage de la meilleure solution de la génération
        best_individual, best_fitness = fitness_scores[0]
        print(f"Génération {generation}: Meilleure fitness = {best_fitness}")
        if best_fitness == 0:
            print("Solution optimale trouvée !")
            break

    # Retourner la meilleure solution trouvée
    return best_individual, best_fitness

# ----------- MAIN ------------

if __name__ == "__main__":
    print("=== Lancement du script ===")
    # Paramètres
    nb_routes = 5
    nb_levels = 3
    period = 10
    c = 2
