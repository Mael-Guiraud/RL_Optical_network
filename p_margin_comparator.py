import numpy as np
import pandas as pd
import time

from p_lib_graph import plot_instance
from network import run_one_network, generate_instance_and_traffic
import copy

# ====== FONCTIONS COMPARATIVES ======

def run_modes(instance, Fs, weight_matrix, nb_routes, period, c, B, modes):
    results = {}
    for mode_label in modes:
        print(f"\n=== Launching run_one_network in mode '{mode_label}' ===")
        start_time = time.time()

        # Call your main function with the chosen mode
        result = run_one_network(
            instance=instance,
            Fs=Fs,
            weight_matrix=weight_matrix,
            nb_routes=nb_routes,
            period=period,
            c=c,
            B=B,
            mode=mode_label
        )

        elapsed_time = time.time() - start_time
        results[mode_label] = (result, elapsed_time)

        # print(f"Result ({mode_label}): {result}")
        # print(f"Computation time: {elapsed_time:.4f} seconds")

    print(f"================================")
    for mode_label, (result, elapsed_time) in results.items():
        print(f"Result ({mode_label}): {result}")
        print(f"Computation time: {elapsed_time:.4f} seconds")
    return results

def run_and_compare_modes(nb_routes, nb_levels, period, c, modes, num_runs=100):
    """
    Runs each specified mode num_runs times and compares the results.

    Args:
        nb_routes: The number of routes.
        nb_levels: The number of levels.
        period: The period length.
        c: The minimal separation.
        modes: A list of modes to run.
        num_runs: The number of times to run each mode.

    Returns:
        A dictionary containing the results for each mode, including average
        success ratio, average execution time, and all individual run results.
    """

    results = {}
    for mode in modes:
        results[mode] = {'ratios': [], 'times': []}

    for i in range(num_runs):  # Outer loop for instances
        print(f"\n=== Running instance {i + 1}/{num_runs} ===")
        instance, Fs, weight_matrix, B = generate_instance_and_traffic(nb_routes, nb_levels, period, c)
        initial_Fs = copy.deepcopy(Fs)  # Store initial Fs

        for mode in modes:  # Inner loop for modes
            print(f"\n--- Running mode '{mode}' ---")
            Fs = copy.deepcopy(initial_Fs)  # Reset before each mode run
            start_time = time.time()
            ratio = run_one_network(instance, Fs, weight_matrix, nb_routes, period, c, B, mode)
            elapsed_time = time.time() - start_time

            results[mode]['ratios'].append(ratio)
            results[mode]['times'].append(elapsed_time)

    # Aggregate results
    for mode in modes:
        successful_runs = [r for r in results[mode]['ratios'] if r != -1]
        avg_ratio = np.mean(successful_runs) if successful_runs else -1  # Avoid error if no successful runs
        avg_time = np.mean(results[mode]['times'])
        results[mode]['avg_ratio'] = avg_ratio
        results[mode]['avg_time'] = avg_time

    # Print aggregated results
    print("\n=== Aggregated Results ===")
    for mode in modes:
        print(f"Mode: {mode}")
        print(f"  Average MRD: {results[mode]['avg_ratio']}")
        print(f"  Average Time: {results[mode]['avg_time']:.4f} seconds")
        # print(f"  All MRD: {results[mode]['ratios']}")  # For detailed analysis
        # print(f"  All Times: {results[mode]['times']}")

    return results

def vary_parameters_and_run(nb_routes_range, nb_levels_range, period_range, c_range, modes, num_runs=10):
    """Varies the given parameters and runs the specified modes for each combination."""

    results = []

    for nb_routes in nb_routes_range:
        for nb_levels in nb_levels_range:
            for period in period_range:
                for c in c_range:
                        # Create a unique key for this parameter combination
                        #params_key = (nb_routes, nb_levels, period, c)
                        #results[params_key] = run_and_compare_modes(nb_routes, nb_levels, period, c, B, modes, num_runs)
                        mode_results = run_and_compare_modes(nb_routes, nb_levels, period, c, modes, num_runs)
                        for mode, result in mode_results.items():
                            results.append({
                                'Route': nb_routes,
                                'Levels': nb_levels,
                                'Period': period,
                                'c': c,
                                'Mode': mode,
                                'Average Ratio': result['avg_ratio'],
                                'Average Time': result['avg_time']
                            })
                        #print("==" * 20)
                        #print(f"==Results for the parameter: {params_key}==")
                        #print(results[params_key])
                        #print("==" * 20)
    return results

def results_to_dataframe(results):
    """Converts the results to a Pandas DataFrame."""
    df = pd.DataFrame(results)
    return df

# ====== MAIN MAIS FLEMME DE FAIRE DU MAIN ======

def start_CompareParametersandprinttoCSV():
    # Define the parameter ranges
    nb_routes_range = [5, 9]
    nb_levels_range = [3, 5]
    period_range = [10, 20]
    c_range = [2, 5]
    # B_range = [3, 5, 7, 9]
    modes = ['POLYOPTI', 'BINARY_SEARCH', 'POLY_INCREMENTAL']
    num_run = 100
    # Run the vary_parameters_and_run function
    results = vary_parameters_and_run(nb_routes_range, nb_levels_range, period_range, c_range, modes, num_run)

    # Convert the results to a DataFrame
    df = results_to_dataframe(results)

    # Display the DataFrame
    df.to_csv('results_margin_comparator.csv', index=False)

def start_CompareModes():
    # Paramètres
    nb_routes = 5
    nb_levels = 3
    period = 10
    c = 2

    modes_to_run = ["POLY", "POLYOPTI", "BINARY_SEARCH", "POLY_HYBRID", "POLY_INCREMENTAL"]
    run_and_compare_modes(nb_routes, nb_levels, period, c, modes_to_run)

def start_RunOneNetwork():
    # Paramètres
    nb_routes = 5
    nb_levels = 3
    period = 10
    c = 2

    # Génération de l'instance et du trafic
    instance, Fs, weight_matrix, B = generate_instance_and_traffic(nb_routes, nb_levels, period, c)
    print("=== Instance générée ===")
    print("Structure:", instance)

    plot_instance(instance)

    # Lancement d'une unique résolution par exemple en mode "POLY"
    print("\n=== Lancement de run_one_network en mode 'POLY LINEAR' ===")
    start_time_poly = time.time()
    poly_value = run_one_network(instance, Fs, weight_matrix, nb_routes, period, c, B, mode="POLY")
    computation_time_poly = time.time() - start_time_poly

    print(f"Résultat (POLY) : {poly_value}")
    print(f"Temps de calcul : {computation_time_poly:.4f} secondes")

if __name__ == "__main__":
    # start_RunOneNetwork()

    #instance,Fs,weight_matrix,B = generate_instance_and_traffic(5,3,10,2)

    #print("=== Parameters ===")
    #print("Instance:", instance)
    #print("Scheduling matrix:", Fs)
    #print("Weight matrix:", weight_matrix)
    #print("Buffer B:", B)

    #start_RunOneNetwork()
    start_CompareParametersandprinttoCSV()