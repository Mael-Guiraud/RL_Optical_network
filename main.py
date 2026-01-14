import duckdb
import numpy as np
import pandas as pd

from p_analysis import compare_all_multiprocessed, concatenate_parquet_files
from p_dataset import get_instance_from_dataset
from p_metaheuristics import recuit_simule, recuit_simule_packet
from p_supervisor import run_one_network_with_margin_extraction, run_network_with_route_supervisor, track_all_packets, packets_matrix_builder


def concat():
    r_routes = [5, 9, 13]
    r_levels = [3, 5]
    r_period = [10, 50, 100]
    folder_path = "./results/all"

    for nb_routes in r_routes:
        for nb_levels in r_levels:
            for period in r_period:
                for c in [2, period / 2]:
                    concatenate_parquet_files(folder_path, nb_routes, nb_levels, period, int(c))

def verif(nb_routes, nb_levels, period, c, inst_id):
    data = get_instance_from_dataset(nb_routes,nb_levels,period,c, inst_id)
    instance, S, Fs, weight_matrix, B = data[4], data[5], data[6], data[7], data[8]
    print(f"Instance : {instance}")
    print(f"S : {S}")
    print(f"Fs : {Fs}")
    print(f"weight_matrix : {weight_matrix}")
    print(f"B : {B}")

    mdr_initial, margin_matrix, e1  = run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, c, B)
    mdr_route, e2 = run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, margin_matrix)
    best_supervisor, mdr_recuit, T, best_error = recuit_simule(instance, Fs, weight_matrix,  nb_routes, period, c, B, margin_matrix, cooling_rate=0.99)
    #print(best_supervisor)

    mdr_route_poly, _ = run_network_with_route_supervisor(instance, Fs, weight_matrix,  nb_routes, period, c, B, best_supervisor)
    print(f"INITIAL : {mdr_initial}")
    print(f"ERROR CODE : {e1}")
    print(f"MARGIN MATRIX : {mdr_route}")
    print(f"ERROR MATRIX: {e2}")
    print(f"RECUIT SIMULE : {mdr_recuit}")
    print(f"ERROR RECUIT : {best_error}")
    print(f"POLYOPTI AVEC MATRICE DU RECUIT : {mdr_route_poly}")


def count_instances():
    r_routes = range(5, 21)
    r_levels = range(2, 5 + 1)
    r_period = range(10, 100 + 1, 10)

    i=0
    for nb_routes in r_routes:
        for nb_levels in r_levels:
            for period in r_period:
                for c in [1, 2, 3, period / 2]:
                    i+=1
    print(i)

def example_pfe():
    data = get_instance_from_dataset(5, 3, 10, 2, 2)
    instance, S, Fs, weight_matrix, B = data[4], data[5], data[6], data[7], data[8]
    print(f"Instance : {instance}")
    # print(f"S : {S}")
    print(f"Fs : {Fs}")
    # print(f"weight_matrix : {weight_matrix}")
    # print(f"B : {B}")
    # plot_instance(instance)
    # t = track_all_packets(Fs, S, weight_matrix, instance, 10, 3)
    # plot_instance_with_tracked_packets(instance, t, base_curvature=0.2, max_curvature=0.3)

    print(f"Fs[0] : {Fs[0]}")
    count_ones = 0

    print("Number of 1s in the matrix:", count_ones)
    all_packets = track_all_packets(Fs, S, weight_matrix, instance, 10, 3)
    nb_paquets = 0
    for route_id, packets in all_packets.items():
        nb_paquets += len(packets)
    print(nb_paquets)

if __name__ == "__main__":
    r_routes = [5]
    r_levels = [2]
    r_period = [10]
    r_c = [2]
    B = [3]
    r_instances = range(0, 100)
    for i in range(100):
        compare_all_multiprocessed(r_routes, r_levels, r_period, r_c, B, r_instances)


    # nb_routes = 5
    # nb_levels = 2
    # period = 10
    # c = 2
    # B = 3
    # inst_id = 0
    #
    # data = get_instance_from_dataset(nb_routes, nb_levels, period, c, inst_id)
    # instance, S, Fs, weight_matrix = data[4], data[5], data[6], data[7]
    #
    # margin_data = packets_matrix_builder(instance, Fs, S, weight_matrix, nb_routes, period)
    # margin_matrix = margin_data['margin_matrix']
    # print(f"margin matrix : {margin_matrix}")
    #
    # best_matrix, best_mrd, T_finale, error_code, solutions = recuit_simule_packet(
    #                 instance, Fs, weight_matrix, nb_routes, period, c, B,
    #                 margin_matrix,
    #                 temperature_threshold=0.01,
    #                 initial_temperature=1,
    #                 cooling_rate=0.98,
    #                 noise=0.1,
    #                 n_routes=1,
    #                 n_packets=1,
    #                 n_levels=1,
    #                 mode="palier"
    #             )
    # print(f"best matrix : {best_matrix}")
    # print(f"best mrd : {best_mrd}")
    # print(f"T finale : {T_finale}")
    # print(f"error code : {error_code}")
    # print(f"solutions : {solutions}")

   # mrd_initial, supervisor_matrix, _ = run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, int(c), B)
   # recuit_simule(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix, mode="palier")
   #
   #  T = 1.0
   #  temperature_threshold = 0.01
   #  cooling_rate = 0.98
   #
   #  i = 0
   #  while T > temperature_threshold:
   #      i += 1
   #      T *= cooling_rate
   #
   #  print(f"Temperature threshold reached after {i} iterations")
    #verif(10,3,20,2,46)

    #count_instances()
    #verif(18,4,10,5,50)

    #
    # générer résultats avec ça
    # 10 ; 3 ; 10 -> 100 ; 5
