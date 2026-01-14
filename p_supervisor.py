import csv
import os

import pulp
import lb_topo
import lb_traffic
from lb_traffic import *
from logging import info
from network import reset_schedule, run_test_case
import numpy as np

# ========== Traffic generation functions ==========

def generate_random_traffic_no_save_bis(instance,period,nb_routes,mini_buff):
    """
    Generates random traffic for a given network instance without saving it to a file.

    This function initializes matrices for traffic flow, scheduling, delays, weights, and buffers.
    It then schedules the traffic using the `schedule_all_routes` function and checks if the
    generated traffic meets certain criteria (e.g., no scheduling conflicts, valid delays).

    Args:
        instance (list): A nested list representing the network topology.
        period (int): The time period for scheduling.
        nb_routes (int): The number of routes in the network.
        mini_buff (int): The minimum buffer time.

    Returns:
        tuple: A tuple containing the scheduling matrix (S_matrix), flow matrix (Fs), weight matrix,
               delay matrix, sum of delays, maximum delay ratio, and maximum buffer size,
               or None if an error occurs during traffic generation.
    """
    instance.insert(0,[[[int(i)]] for i in range(nb_routes)])
    Fs = init_all_node_matrix(instance,period,nb_routes)
    S_matrix = init_all_node_matrix(instance,period,nb_routes)
    delays = init_all_node_matrix(instance,period,nb_routes)
    weight_matrix = init_weight_matrix(instance,period)
    buffer_matrix = init_buffer_matrix(instance,period)
    remaining_tics = init_remaining_tics_matrix(instance,period)
    non_scheduled_routes = init_non_scheduled_routes(instance,period)

    schedule_all_routes(instance,S_matrix,buffer_matrix,weight_matrix,Fs,delays,remaining_tics,non_scheduled_routes,period,nb_routes,mini_buff,0.5)

    if (check_FS(Fs)
            and check_FS2(Fs)
            and check_FS3(Fs,nb_routes)
            and check_delays(delays,mini_buff)):
        return S_matrix, Fs,weight_matrix,delays,sum_delay(delays),max_ratio_delay(delays,weight_matrix,instance),int(find_max_buffer(buffer_matrix))
    print("Error in the generation of the traffic")
    return None

def generate_instance_and_traffic_bis(nb_routes,nb_levels,period,c):
    """
    Generates a network instance and random traffic for it.

    This function repeatedly attempts to generate a valid network instance using
    `lb_topo.generate_instance` until a valid instance is found. It then generates
    random traffic for this instance using `generate_random_traffic_no_save_bis`.

    Args:
        nb_routes (int): The number of routes in the network.
        nb_levels (int): The number of levels in the network.
        period (int): The time period for scheduling.
        c (int): The minimum buffer time.

    Returns:
        tuple: A tuple containing the network instance, scheduling matrix (S), flow matrix (Fs),
               weight matrix, and maximum buffer size (B).
    """
    instance = None
    while not instance:
        instance,nb_fails= lb_topo.generate_instance(nb_routes, nb_levels)
    if instance:
        #print(lb_topo.check_weakly_coherent(instance,nb_routes),lb_topo.check_strongly_coherent(instance,nb_routes))
        S,Fs,weight_matrix,_,_,_,B = generate_random_traffic_no_save_bis(instance, period, nb_routes, c)
    else:
        print("Error, no instance found")
        exit(1)
    return instance,S,Fs,weight_matrix,B

# ========== Supervisor functions ==========

def network_error_manager(Fs,nb_routes,delays,c,S,weight_matrix,period,instance,B,mode,F):
    erreurs = False
    n = 0
    if not lb_traffic.check_FS(Fs):
        n=1
        erreurs = True
    if not lb_traffic.check_FS2(Fs):
        n=2
        erreurs = True
    if not lb_traffic.check_FS3(Fs, nb_routes):
        n=3
        erreurs = True
    if not lb_traffic.check_delays(delays, c):
        n=4
        erreurs = True
    if not lb_traffic.check_matrices_Fs_delays(Fs, delays):
        n=5
        erreurs = True
    if not lb_traffic.check_if_pattern_ok(S, Fs, weight_matrix, period, instance):
        n=6
        erreurs = True
    if not lb_traffic.check_buffer_min(S, c):
        n=7
        erreurs = True
    if not lb_traffic.check_buffer_max(S, Fs, weight_matrix, period, instance, B):
        n=8
        erreurs = True
    if erreurs:
        write_to_csv('./results/errors.csv',
                     n,
                     mode,
                     F,
                     instance,
                     weight_matrix,
                     B,
                     Fs,
                     delays,
                     S)
    return n

def run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, c, B, solver_mode="POLYOPTI"):
    """
    Runs the network scheduling while recording the margin used for each route at every level.
    Returns the max ratio delay and a supervisor matrix that has one row per route
    and one column per level (excluding the starting level) containing the margin percentage.

    The margin percentage is computed as:
        percentage = (final_margin / max_margin)
    where max_margin is typically (period + c).

    Args:
        instance (list): The network instance.
        Fs (list): The flow matrix.
        weight_matrix (list): The weight matrix.
        nb_routes (int): The number of routes.
        period (int): The time period.
        c (int): The minimum buffer time.
        B (int): The buffer size.
        solver_mode (str): The operational mode ("POLY" or "POLYOPTI").
        matrix (str): The type of matrix to return ("classic" or "normalized").

    Returns:
        tuple: A tuple containing the maximum delay ratio and the supervisor matrix.
    """
    # Pre-calculate minimal route lengths.
    taumin = [lb_traffic.route_length(rid, weight_matrix, instance) for rid in range(nb_routes)]

    F = Fs[0]

    reset_schedule(Fs)
    S = lb_traffic.init_all_node_matrix(instance, period, nb_routes)
    delays = lb_traffic.init_all_node_matrix(instance, period, nb_routes)

    nb_levels = len(Fs)
    # Create a matrix to record the final margin used for each route at each level (levels 1..nb_levels-1).
    # For route r, route_margin_matrix[r][l] will hold the margin used at level (l+1) (since level 0 is the start).
    route_margin_matrix = [[None for _ in range(nb_levels - 1)] for _ in range(nb_routes)]

    # Process each level (skipping level 0, the starting node)
    for i, level in enumerate(Fs):
        if i > 0:
            for j, router in enumerate(level):  # For each router in the current level.
                for k, cycle in enumerate(router):  # For each cycle in the current router.
                    if solver_mode in ("POLY", "POLYOPTI"):
                        found = False
                        margin = 0
                        # Try increasing the margin until a valid assignment is found or margin reaches the maximum.
                        while not found and margin < period:
                            tasks = []
                            # For each route in the current node, build the tasks with the current margin.
                            for l in instance[i][j][k]:
                                previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, l)
                                for t in range(period):
                                    if Fs[i - 1][previous_r][previous_cycle][l][t] == 1:
                                        r_i = (t + weight_matrix[i - 1][previous_r][previous_cycle]) % period
                                        d = delays[i - 1][previous_r][previous_cycle][l][t]
                                        d_i = r_i + margin
                                        tasks.append({
                                            'id': f'task_{l}_{r_i}',
                                            'r_i': r_i,
                                            'd_i': d_i,
                                            'delay': d,
                                            'tau': taumin[l]
                                        })
                            if len(tasks) == 0:
                                print("Error, no task found")
                                exit(1)
                            # If there is only one task, assign it directly.
                            if len(tasks) == 1:
                                parts = tasks[0]['id'].split('_')
                                route_id = int(parts[1])
                                time = int(parts[2])
                                Fs[i][j][k][route_id][time] = 1
                                # Record the margin for this route at level i (stored in column index i-1).
                                route_margin_matrix[route_id][i - 1] = margin
                                found = True
                                previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, route_id)
                                delays[i][j][k][route_id][time] = delays[i - 1][previous_r][previous_cycle][route_id][(time - weight_matrix[i - 1][previous_r][previous_cycle]) % period]


                            else:
                                # Multiple tasks: solve the one-node problem.
                                if solver_mode == "POLYOPTI":
                                    result = run_test_case(nb_routes, tasks, period, c, B, pulp.PULP_CBC_CMD(msg=False),"opti")
                                else:
                                    result = run_test_case(nb_routes, tasks, period, c, B, pulp.PULP_CBC_CMD(msg=False),"basic")

                                # If not all tasks could be assigned, try a higher margin.
                                if len(result["assigned_dates"]) != len(tasks):
                                    margin += 1
                                    found = False
                                else:
                                    # A valid assignment was found: update Fs, delays, S and record the margin.
                                    for key, value in result["assigned_dates"].items():
                                        parts = key.split('_')
                                        route_id = int(parts[1])
                                        time = int(parts[2])
                                        new_time = value
                                        # Compute the delay (taking wrap-around into account).
                                        delai = (new_time - time) % period if (((new_time - time) % period) >= c) or (((new_time - time) % period) == 0) else (new_time - time) % period + period
                                        previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, route_id)
                                        Fs[i][j][k][route_id][new_time] = 1
                                        # delays[i][j][k][route_id][new_time] = delays[i - 1][previous_r][previous_cycle][route_id][time] + delai
                                        #delays[i][j][k][route_id][new_time] = delays[i - 1][previous_r][previous_cycle][route_id][time - weight_matrix[i - 1][previous_r][previous_cycle]] + delai
                                        delays[i][j][k][route_id][new_time] = delays[i - 1][previous_r][previous_cycle][route_id][(time - weight_matrix[i - 1][previous_r][previous_cycle]) % period] + delai
                                        S[i][j][k][route_id][time] = delai
                                        # Record the margin used for this route at this level.
                                        route_margin_matrix[route_id][i - 1] = margin
                                    found = (not result["error"] and result["status"] != "Infeasible")
                                    #print(result["error_details"], result["status"])
                                    # Increment margin for the next attempt (if any).
                                    margin += 1
                        if not found:
                            print("Error, no solution found while margin is maximum")
                            print(result["error_details"], result["status"])
                            print("Instance:", instance)
                            print("Fs matrix:", Fs)
                            print("Weight matrix:", weight_matrix)
                            print("B:", B)
                            print("C",c)
                            print("Route margin matrix:", route_margin_matrix)
                            print("Delay matrix:", delays)
                            print("S matrix:", S)
                            print("Period:", period)
                            print("Mode:", solver_mode)
                            return -1, None, 10

    # Verification of schedules/delays
    e = network_error_manager(Fs, nb_routes, delays, c, S, weight_matrix, period, instance, B, solver_mode, F)

    # After processing, convert the absolute margin values into percentages.
    max_margin = period + c
    supervisor_matrix = []

    for r in range(nb_routes):
        row = [(m if m is not None else 0) / max_margin for m in route_margin_matrix[r]]
        total = sum(row)
        if total != 0:
            row = [value / total for value in row]
        supervisor_matrix.append(row)
    # Return both the performance measure and the supervisor matrix.

    lb_traffic.max_ratio_delay(delays, weight_matrix, instance)

    return lb_traffic.max_ratio_delay(delays, weight_matrix, instance), supervisor_matrix, e

def run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix, solver_mode="POLYOPTI", max_margin=None):
    """
    Run the network with supervisor guidance and calculate scheduling and delays.

    This function takes a network model with associated parameters and computes the task scheduling
    and delay values for the network. It iterates over the levels, routers, and cycles of the
    given network instance, implements supervisor margin allocation, and assigns schedules and delays
    based on the given parameters and operational modes. The function works with optimization routines
    to manage task assignments and ensures the resulting schedules and delays abide by specific constraints.

    Args:
        instance (list): Representation of the network model, where each element typically corresponds
            to a level in the network that contains routing information.
        Fs (list): A list that represents the scheduling matrix for different levels in the network.
        weight_matrix (list): A matrix defining routing delays and weights for transitions between nodes or cycles.
        nb_routes (int): The number of routes in the network.
        period (int): The periodicity of the network operations, defining the cycle length.
        c (int): A constraint parameter representing a minimum required delay or margin value.
        B (int): A constraint parameter typically representing buffer capacity or similar restriction in the network.
        solver_mode (str): The operational mode for task assignment, e.g., optimization ("POLYOPTI") or basic mode.
        supervisor_matrix (list): A matrix where each element represents a percentage value allocated to a
            route, defining its supervision margin at each level.
        max_margin (int, optional): The maximum margin of delay allowed. If None, it defaults to `period + c - 1`.

    Returns:
        float: Maximum ratio delay calculated across all routes.

    Raises:
        SystemExit: If no valid task is found in the task scheduling process, causing the program to terminate.
    """

    if max_margin is None:
        max_margin = period
        # tester avec period / 2, period, et period * 2

    if supervisor_matrix is None:
        # récupérer une margin de base équitable quand vide
        supervisor_matrix = generate_uniform_supervisor_matrix(instance, nb_routes)

    # Compute minimal task lengths per route
    taumin = [lb_traffic.route_length(rid, weight_matrix, instance) for rid in range(nb_routes)]

    F = Fs[0]
    reset_schedule(Fs)
    S = lb_traffic.init_all_node_matrix(instance, period, nb_routes)
    delays = lb_traffic.init_all_node_matrix(instance, period, nb_routes)

    # Iterate over nodes (levels, routers, cycles)
    for i, level in enumerate(Fs):
        if i > 0:
            for j, router in enumerate(level):
                for k, cycle in enumerate(router):
                    tasks = []
                    # Instead of computing margin once per node, loop over each route id in this node.
                    for route in instance[i][j][k]:
                        # Map the current level to the supervisor matrix column (assuming level 0 is omitted)
                        supervisor_percentage = supervisor_matrix[route][i - 1]
                        margin = int(round(supervisor_percentage * max_margin))
                        #print(f"Supervisor margin allocation of {margin} (percentage {supervisor_percentage}) for route {route} at level {i} (node ({j},{k}))")

                        previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, route)
                        for t in range(period):
                            if Fs[i - 1][previous_r][previous_cycle][route][t] == 1:
                                r_i = (t + weight_matrix[i - 1][previous_r][previous_cycle]) % period
                                d = delays[i - 1][previous_r][previous_cycle][route][t]
                                d_i = r_i + margin
                                tasks.append({
                                    'id': f'task_{route}_{r_i}',
                                    'r_i': r_i,
                                    'd_i': d_i,
                                    'delay': d,
                                    'tau': taumin[route],
                                    'margin': margin  # optional, for clarity
                                })
                    if len(tasks) == 0:
                        print("Error, no task found")
                        exit(1)

                    # When only one task exists, simply assign it.
                    if len(tasks) == 1:
                        parts = tasks[0]['id'].split('_')
                        route_id = int(parts[1])
                        time = int(parts[2])
                        Fs[i][j][k][route_id][time] = 1

                        previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, route_id)
                        delays[i][j][k][route_id][time] = delays[i - 1][previous_r][previous_cycle][route_id][(time - weight_matrix[i - 1][previous_r][previous_cycle]) % period]

                    else:
                        if solver_mode == "POLYOPTI":
                            result = run_test_case(nb_routes, tasks, period, c, B, pulp.PULP_CBC_CMD(msg=False), "opti")
                        else:
                            result = run_test_case(nb_routes, tasks, period, c, B, pulp.PULP_CBC_CMD(msg=False),"basic")

                        if len(result["assigned_dates"]) != len(tasks):
                            #print(f"Supervisor margin allocation failed for node ({i},{j},{k})")
                            # Return a penalty value to indicate that this solution is infeasible.
                            return float("inf"), 10
                        else:
                            for key, value in result["assigned_dates"].items():
                                parts = key.split('_')
                                route_id = int(parts[1])
                                time = int(parts[2])
                                new_time = value
                                delai = (new_time - time) % period if (((new_time - time) % period) >= c) or (((new_time - time) % period) == 0) else (new_time - time) % period + period
                                previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1,route_id)
                                Fs[i][j][k][route_id][new_time] = 1
                                # delays[i][j][k][route_id][new_time] = delays[i-1][previous_r][previous_cycle][route_id][time-weight_matrix[i-1][previous_r][previous_cycle]] + delai
                                delays[i][j][k][route_id][new_time] = delays[i-1][previous_r][previous_cycle][route_id][ (time-weight_matrix[i-1][previous_r][previous_cycle]) % period] + delai
                                # if route_id == 3:
                                #     #print("Delay is not 0")
                                #     print(delays[i][j][k][route_id][new_time])
                                #     print(delays[i-1][previous_r][previous_cycle][route_id][ (time-weight_matrix[i-1][previous_r][previous_cycle]) % period] + delai)
                                #     print([i-1],[previous_r],[previous_cycle],[route_id],weight_matrix[i-1],[previous_r],[previous_cycle], delai)
                                #     print(f"{time}")
                                #     print(f"{new_time}")

                                S[i][j][k][route_id][time] = delai

    # Verification of schedules and delays
    e = network_error_manager(Fs, nb_routes, delays, c, S, weight_matrix, period, instance, B, solver_mode, F)

    #print(f"S : {S}")
    #print(f"DELAYS : {delays}")

    # Determine max delay on the final level for reporting
    """ 
    max_delay = 0
    route_max_delay = -1
    for j, router in enumerate(delays[-1]):
        for k, cycle in enumerate(router):
            for l, route in enumerate(cycle):
                for m, tic in enumerate(route):
                    if delays[-1][j][k][l][m] > max_delay:
                        max_delay = delays[-1][j][k][l][m]
                        route_max_delay = l
    info("Max delay:", max_delay, " on route ", route_max_delay)
    print("tau_min:", taumin)
    """
    return lb_traffic.max_ratio_delay(delays, weight_matrix, instance), e

def generate_uniform_supervisor_matrix(instance, nb_routes):
    """
    Génère une matrice de supervision avec une distribution uniforme des marges pour chaque route.

    Cette fonction crée une matrice où chaque ligne représente une route et chaque colonne représente
    un niveau dans le réseau (à l'exception du niveau de départ). Les valeurs de la matrice sont les
    pourcentages de marge alloués à chaque niveau, avec une distribution uniforme.

    Args:
        instance (list): L'instance du réseau, représentant la topologie.
        nb_routes (int): Le nombre de routes dans le réseau.

    Returns:
        list: Une matrice de supervision où supervisor_matrix[r][i] contient le pourcentage
              de marge alloué à la route r au niveau i+1 (car le niveau 0 est le départ).
              La somme des valeurs pour chaque route est égale à 1.
    """
    # Déterminer le nombre de niveaux dans le réseau
    nb_levels = len(instance)

    # Pour une distribution uniforme, chaque niveau reçoit la même proportion
    # Nous excluons le niveau 0 (le niveau de départ) qui n'a pas besoin de marge
    supervisor_matrix = []

    if nb_levels <= 1:
        # Cas particulier: réseau avec un seul niveau
        return [[0] for _ in range(nb_routes)]

    # Distribution uniforme: chaque niveau reçoit 1/(nb_levels-1) de la marge totale
    uniform_value = 1.0 / (nb_levels - 1)

    # Créer la matrice de supervision
    for r in range(nb_routes):
        # Une ligne par route, avec une colonne par niveau (sauf le niveau 0)
        row = [uniform_value for _ in range(nb_levels - 1)]
        supervisor_matrix.append(row)

    return supervisor_matrix

# ========= Packets supervisor functions ==========

def track_packets(Fs, S, weight_matrix, instance, period, route_id):
    """
    Track all packets of a specific route through the network.

    Parameters:
    -----------
    Fs : List[List[List[List[List[int]]]]]
        The scheduling matrix (levels -> routers -> cycles -> routes -> time slots)
    S : List[List[List[List[List[int]]]]]
        The delay matrix
    weight_matrix : List[List[List[int]]]
        Matrix of weights between levels
    instance : List[List[List[int]]]
        Network instance structure
    period : int
        Period length
    route_id : int
        ID of the route to track

    Returns:
    --------
    List[Dict]
        List of dictionaries containing packet information:
        - packet_id: Packet index (0, 1, ...)
        - initial_time_slot: Initial time slot at level 0
        - path: List of tuples (level, router, cycle, time_slot) showing packet's path
    """
    packets = []

    # Find all packets at level 0
    initial_time_slots = []
    for time_slot in range(period):
        if Fs[0][route_id][0][route_id][time_slot] == 1:
            initial_time_slots.append(time_slot)
    for idx, time_slot in enumerate(initial_time_slots):
        packet = {
            'packet_id': idx,  # Now a simple index
            'initial_time_slot': time_slot,
            'path': [(0, route_id, 0, time_slot)]
        }
        packets.append(packet)

    # Track each packet through subsequent levels
    for packet in packets:
        current_level = 0
        current_router = route_id
        current_cycle = 0
        current_time = packet['initial_time_slot']

        # Follow packet through each level
        while current_level < len(instance) - 1:
            # Get weight for current position
            weight = weight_matrix[current_level][current_router][current_cycle]
            if isinstance(weight, (list, np.ndarray)):
                weight = weight[0]  # Take first element if it's a list/array

            # Find next router and cycle for this route
            next_router, next_cycle = find_id_router_and_cycle(instance, current_level + 1, route_id)

            # Calculate arrival time at next level
            arrival_time = (current_time + weight) % period
            # Add delay from S matrix
            delay = int(S[current_level + 1][next_router][next_cycle][route_id][arrival_time])
            next_time = (arrival_time + delay) % period
            
            # Add this position to packet's path
            packet['path'].append((current_level + 1, next_router, next_cycle, next_time))
            
            # Update current position for next iteration
            current_level += 1
            current_router = next_router
            current_cycle = next_cycle
            current_time = next_time
    
    return packets

def track_all_packets(Fs, S, weight_matrix, instance, period, nb_routes):
    """
    Track all packets for all routes through the network.
    
    Parameters:
    -----------
    Fs : List[List[List[List[List[int]]]]]
        The scheduling matrix (levels -> routers -> cycles -> routes -> time slots)
    S : List[List[List[List[List[int]]]]]
        The delay matrix
    weight_matrix : List[List[List[int]]]
        Matrix of weights between levels
    instance : List[List[List[int]]]
        Network instance structure
    period : int
        Period length
    nb_routes : int
        Number of routes in the network
        
    Returns:
    --------
    Dict[int, List[Dict]]
        Dictionary mapping route_id to a list of its packets, where each packet is a dictionary:
        - packet_id: Packet index (0, 1, ...)
        - initial_time_slot: Initial time slot at level 0
        - path: List of tuples (level, router, cycle, time_slot) showing packet's path
    """
    all_packets = {}
    
    # Track packets for each route
    for route_id in range(nb_routes):
        route_packets = track_packets(Fs, S, weight_matrix, instance, period, route_id)
        all_packets[route_id] = route_packets
    
    return all_packets

def packets_matrix_builder(instance, Fs, S, weight_matrix, nb_routes, period):
    """
    Compte le nombre de paquets par route et établit une matrice d'attribution de margin pour chaque paquet.
    
    Parameters:
    -----------
    instance : List[List[List[int]]]
        Network instance structure
    Fs : List[List[List[List[List[int]]]]]
        The scheduling matrix (levels -> routers -> cycles -> routes -> time slots)
    weight_matrix : List[List[List[int]]]
        Matrix of weights between levels
    nb_routes : int
        Number of routes in the network
    period : int
        Period length
        
    Returns:
    --------
    Dict
        Dictionary containing:
        - 'packet_counts': Dictionary mapping route_id to number of packets
        - 'margin_matrix': Matrix of margin attribution for each packet
    """
    # Obtenir tous les paquets pour toutes les routes
    all_packets = track_all_packets(Fs, S, weight_matrix, instance, period, nb_routes)
    
    # Compter le nombre de paquets par route
    packet_counts = {route_id: len(packets) for route_id, packets in all_packets.items()}
    
    # Créer une matrice d'attribution de margin pour chaque paquet
    nb_levels = len(instance)
    margin_matrix = {}
    
    # Pour chaque route
    for route_id, packets in all_packets.items():
        route_margin = []
        
        # Pour chaque paquet de cette route
        for packet in packets:
            # Créer un vecteur de margin pour chaque niveau (sauf le niveau 0)
            packet_margin = [0] * (nb_levels - 1)
            
            # Distribution uniforme de la margin pour ce paquet
            if nb_levels > 1:
                uniform_margin = 1.0 / (nb_levels - 1)
                packet_margin = [uniform_margin] * (nb_levels - 1)
            
            route_margin.append(packet_margin)
        
        margin_matrix[route_id] = route_margin
    
    return {
        'packet_counts': packet_counts,
        'margin_matrix': margin_matrix
    }

def packets_run_network(instance, Fs, weight_matrix, nb_routes, period, c, B, margin_matrix, max_margin=None):
    """
    Simulates the execution of packets through a network using specified parameters and configurations.

    This function processes network routes, tasks, and delays for packets traveling through routers at
    different levels in the network. It uses parameters such as weight matrices and margins to calculate
    packet delays and schedules. The function generates an error report and calculates the maximum ratio
    of delays across the network.

    Args:
        instance: The network instance containing details of the routers and routes.
        Fs: The schedule matrix for all levels, nodes, cycles, and packets.
        weight_matrix: The weight matrix defining the traversal weights between routers.
        nb_routes: Total number of routes in the network.
        period: The period for packet transmissions.
        c: The constant delay threshold.
        B: The buffer capacity for nodes.
        margin_matrix: Matrix containing margin percentages for tasks on different routes and levels.
        max_margin: Optional; the maximum allowed margin for tasks. Defaults to period + c.

    Returns:
        tuple: A tuple containing the maximum ratio of delays across the network and the error report.
    """
    if max_margin is None:
        max_margin = period

    # Calculer la longueur minimale de chaque route
    taumin = [lb_traffic.route_length(rid, weight_matrix, instance) for rid in range(nb_routes)]

    # Initialiser les matrices
    reset_schedule(Fs)
    S = lb_traffic.init_all_node_matrix(instance, period, nb_routes)
    delays = lb_traffic.init_all_node_matrix(instance, period, nb_routes)

    # Suivre les paquets au niveau 0
    packets_by_index = {
        (route_id, packet['packet_id']): {
            'route_id': route_id,
            'packet_id': packet['packet_id'],
            'current_level': 0,
            'current_router': route_id,
            'current_cycle': 0,
            'current_time': packet['initial_time_slot']
        }
        for route_id in range(nb_routes)
        for packet in track_packets(Fs, S, weight_matrix, instance, period, route_id)
    }

    # Traiter chaque niveau
    for i in range(1, len(Fs)):
        node_tasks = {}

        # Construire les tâches pour chaque nœud
        for packet_key, packet in list(packets_by_index.items()):
            if packet['current_level'] == i - 1:
                route_id = packet['route_id']
                prev_router, prev_cycle, prev_time = packet['current_router'], packet['current_cycle'], packet['current_time']
                next_router, next_cycle = lb_traffic.find_id_router_and_cycle(instance, i, route_id)
                weight = weight_matrix[i - 1][prev_router][prev_cycle]
                r_i = (prev_time + weight) % period

                # Calculer la marge
                margin_percentage = margin_matrix[route_id][packet['packet_id']][i - 1]
                # print(f"margin_percentage: {margin_percentage}")
                # print(f"max_margin: {max_margin}")
                margin = int(round(margin_percentage * max_margin))

                # Créer la tâche
                task = {
                    'id': f'task_{route_id}_{r_i}',
                    'r_i': r_i,
                    'd_i': r_i + margin,
                    'delay': delays[i - 1][prev_router][prev_cycle][route_id][prev_time],
                    'tau': taumin[route_id],
                    'packet_info': packet
                }
                node_tasks.setdefault((next_router, next_cycle), []).append(task)
                del packets_by_index[packet_key]

        # Résoudre les tâches pour chaque nœud
        for (j, k), tasks in node_tasks.items():
            if len(tasks) == 1:
                task = tasks[0]
                route_id, time = int(task['id'].split('_')[1]), int(task['id'].split('_')[2])
                Fs[i][j][k][route_id][time] = 1
                packets_by_index[(route_id, time)] = {
                    **task['packet_info'],
                    'current_level': i,
                    'current_router': j,
                    'current_cycle': k,
                    'current_time': time
                }

                prev_router, prev_cycle = lb_traffic.find_id_router_and_cycle(instance, i - 1, route_id)
                delays[i][j][k][route_id][time] = delays[i - 1][prev_router][prev_cycle][route_id][(time - weight_matrix[i - 1][prev_router][prev_cycle]) % period]


            else:
                result = run_test_case(nb_routes, tasks, period, c, B, pulp.PULP_CBC_CMD(msg=False), "opti")
                if len(result["assigned_dates"]) != len(tasks):
                    return float("inf"), 10
                for key, value in result["assigned_dates"].items():
                    route_id, time, new_time = int(key.split('_')[1]), int(key.split('_')[2]), value
                    delai = (new_time - time) % period
                    if delai < c and delai != 0:
                        delai += period
                    task = next(t for t in tasks if t['id'] == key)
                    prev_router, prev_cycle, prev_time = task['packet_info']['current_router'], task['packet_info']['current_cycle'], task['packet_info']['current_time']
                    Fs[i][j][k][route_id][new_time] = 1
                    delays[i][j][k][route_id][new_time] = delays[i - 1][prev_router][prev_cycle][route_id][prev_time] + delai
                    #delays[i][j][k][route_id][new_time] = delays[i - 1][prev_router][prev_cycle][route_id][(time - weight_matrix[i - 1][prev_router][prev_cycle]) % period] + delai

                    S[i][j][k][route_id][time] = delai
                    packets_by_index[(route_id, new_time)] = {
                        **task['packet_info'],
                        'current_level': i,
                        'current_router': j,
                        'current_cycle': k,
                        'current_time': new_time
                    }

    # Vérification des erreurs
    e = network_error_manager(Fs, nb_routes, delays, c, S, weight_matrix, period, instance, B, "POLYOPTI", Fs[0])
    return lb_traffic.max_ratio_delay(delays, weight_matrix, instance), e

def write_to_csv(file_path, n, mode, F, instance, weight_matrix, B, Fs, delays, S):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Prepare the data to be written
    data = {
        "Error": n,
        "Mode": mode,
        "F": F,
        "Instance": instance,
        "Weight_Matrix": weight_matrix,
        "B": B,
        "FS": Fs,
        "Delays": delays,
        "S": S
    }

    # Check if the file exists
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())

        # Write the header if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write the data
        writer.writerow(data)