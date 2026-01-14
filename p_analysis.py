import csv
import datetime
import os
import time
import duckdb
import pandas as pd
from pathlib import Path

from lb_traffic import count_ones_in_most_loaded_matrix
from network import generate_instance_and_traffic
from p_dataset import get_instance_from_dataset
from p_metaheuristics import recuit_simule, recuit_simule_packet
from p_supervisor import run_one_network_with_margin_extraction, run_network_with_route_supervisor, packets_matrix_builder, track_all_packets


# ----------- ANALYSIS FUNCTIONS ------------

def compare_metaheuristics(instance, Fs, weight_matrix, nb_routes, nb_levels, period, c, B, initial_mrd, initial_margin_route_matrix, S, iid, folder_path):
    """
    Compare metaheuristics, including recuit_simule_packet, and store results in a Parquet file.

    Args:
        instance, Fs, weight_matrix, nb_routes, period, c, B: Problem parameters
        initial_margin_route_matrix: Initial supervisor matrix
        S: Delay matrix (required for packet-based metaheuristics)
    """
    results = []
    #initial_mrd, e = run_network_with_route_supervisor(instance, Fs, weight_matrix, nb_routes, period, c, B, initial_margin_route_matrix)
    filename = folder_path + f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_b{B}_i{iid}.csv"

    nb_packet = count_ones_in_most_loaded_matrix(Fs)[0]

    r_noise = [0.1]
    r_node = [1]
    r_rows = [1]

    mode = "palier"
    """
    # --- Classic Simulated Annealing ---
    sa_configs = []
    for noise in r_noise:
        for nb_node in r_node:
            for rows in r_rows:
                sa_configs.append({
                    "temp_threshold": 0.01,
                    "init_temp": 1.0,
                    "cooling_rate": 0.98,
                    "noise": noise,
                    "values_per_row": nb_node,
                    "rows": rows,
                    "mode": mode
                })

    i = 0
    for config in sa_configs:
        i += 1
        #print(f"SA {i}/{len(sa_configs)}")
        try:
            start_time = time.time()
            best_supervisor, best_mrd, T_finale, error_code, solutions = recuit_simule(
                instance, Fs, weight_matrix, nb_routes, period, c, B,
                initial_margin_route_matrix,
                temperature_threshold=config["temp_threshold"],
                initial_temperature=config["init_temp"],
                cooling_rate=config["cooling_rate"],
                noise=config["noise"],
                values_per_row=config["values_per_row"],
                rows=config["rows"],
                mode=config["mode"]
            )
            end_time = time.time()
            result = {
                "algorithm": f"SA_{config['mode']}",
                "r": nb_routes,
                "l": nb_levels,
                "p": period,
                "c": c,
                "b": B,
                "instance_id": iid,
                "initial_mrd": initial_mrd,
                "final_mrd": best_mrd,
                "nb_packet": nb_packet,
                "convergence_tempe": T_finale,
                "temp_threshold": config["temp_threshold"],
                "init_temp": config["init_temp"],
                "cooling_rate": config["cooling_rate"],
                "noise": config["noise"],
                "values_per_row": config["values_per_row"],
                "rows": config["rows"],
                "mode": config["mode"],
                "time": end_time - start_time,
                "error_code": error_code,
                "solutions": solutions
            }
        except:
            print("Erreur du recuit ")
            result = {
                "algorithm": f"SA_{config['mode']}",
                "r": nb_routes,
                "l": nb_levels,
                "p": period,
                "c": c,
                "b": B,
                "instance_id": iid,
                "initial_mrd": initial_mrd,
                "final_mrd": -1,
                "nb_packet": nb_packet,
                "convergence_tempe": -1,
                "temp_threshold": config["temp_threshold"],
                "init_temp": config["init_temp"],
                "cooling_rate": config["cooling_rate"],
                "noise": config["noise"],
                "values_per_row": config["values_per_row"],
                "rows": config["rows"],
                "mode": config["mode"],
                "time": -1,
                "error_code": 10,
                "solutions": -1
            }


        results.append(result)
        #save_results_to_parquet(filename, results)
    """
    # --- Packet-based Simulated Annealing ---

    margin_data = packets_matrix_builder(instance, Fs, S, weight_matrix, nb_routes, period)
    margin_matrix = margin_data['margin_matrix']
    packet_configs = []
    for noise in [0.1]:
        for n_routes in [1]:
            for n_packets in [1]:
                for n_levels in [1]:
                    packet_configs.append({
                        "temp_threshold": 0.01,
                        "init_temp": 10.0,
                        "cooling_rate": 0.98,
                        "noise": noise,
                        "n_routes": n_routes,
                        "n_packets": n_packets,
                        "n_levels": n_levels,
                        "mode": mode
                    })
    j = 0
    for config in packet_configs:
        j += 1
        #print(f"Packet SA {j}/{len(packet_configs)}")
        try:
            start_time = time.time()
            best_matrix, best_mrd, T_finale, error_code, solutions = recuit_simule_packet(
                instance, Fs, weight_matrix, nb_routes, period, c, B,
                margin_matrix,
                temperature_threshold=config["temp_threshold"],
                initial_temperature=config["init_temp"],
                cooling_rate=config["cooling_rate"],
                noise=config["noise"],
                n_routes=config["n_routes"],
                n_packets=config["n_packets"],
                n_levels=config["n_levels"],
                mode=config["mode"]
            )
            end_time = time.time()

            result = {
                "algorithm": f"PacketSA_{config['mode']}",
                "r": nb_routes,
                "l": nb_levels,
                "p": period,
                "c": c,
                "b": B,
                "instance_id": iid,
                "initial_mrd": initial_mrd,
                "final_mrd": best_mrd,
                "nb_packet": nb_packet,
                "convergence_tempe": T_finale,
                "temp_threshold": config["temp_threshold"],
                "init_temp": config["init_temp"],
                "cooling_rate": config["cooling_rate"],
                "noise": config["noise"],
                "n_routes": config["n_routes"],
                "n_packets": config["n_packets"],
                "n_levels": config["n_levels"],
                "mode": config["mode"],
                "time": end_time - start_time,
                "error_code": error_code,
                "solutions": solutions
            }
        except:
            #print("Erreur du recuit")
            result = {
                "algorithm": f"PacketSA_{config['mode']}",
                "r": nb_routes,
                "l": nb_levels,
                "p": period,
                "c": c,
                "b": B,
                "instance_id": iid,
                "initial_mrd": initial_mrd,
                "final_mrd": -1,
                "nb_packet": nb_packet,
                "convergence_tempe": -1,
                "temp_threshold": config["temp_threshold"],
                "init_temp": config["init_temp"],
                "cooling_rate": config["cooling_rate"],
                "noise": config["noise"],
                "n_routes": config["n_routes"],
                "n_packets": config["n_packets"],
                "n_levels": config["n_levels"],
                "mode": config["mode"],
                "time": -1,
                "error_code": 10,
                "solutions": -1
            }
        results.append(result)
    # save_results_to_parquet(filename, results)
    save_results_to_csv(filename, results)

# ----------- CSV/PARQUET functions ------------

def save_results_to_csv(filename, results):
    df = pd.DataFrame(results)
    con = duckdb.connect(database=':memory:')
    con.register('df', df)
    con.execute(f"COPY (SELECT * FROM df) TO '{filename}' (HEADER, DELIMITER ',');")

def concatenate_csv_files(folder_path, nb_routes, nb_levels, period, c, B):
    # Define the output file name
    output_file = os.path.join(folder_path, f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_b{B}.csv")

    # Get a list of all files matching the pattern
    input_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.startswith(f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_b{B}_i") and f.endswith(".csv")
    ]

    if not input_files:
        print("No files matching the pattern were found.")
        return

    # Use DuckDB to concatenate the files
    con = duckdb.connect(database=':memory:')

    # Create a table from the first file to get the schema
    con.execute(f"CREATE TABLE results AS SELECT * FROM read_csv_auto('{input_files[0]}')")

    # Insert data from the remaining files
    for file in input_files[1:]:
        con.execute(f"INSERT INTO results SELECT * FROM read_csv_auto('{file}')")

    # Write the concatenated result to a new CSV file
    con.execute(f"COPY results TO '{output_file}' (HEADER, DELIMITER ',');")

    print(f"Concatenated file saved as {output_file}")

    # Optionally delete the original input files
    for file in input_files:
        os.remove(file)
        print(f"{file} removed.")

def save_results_to_parquet(filename, results):
    """
    Saves computational results to a Parquet file using DuckDB for efficient data storage
    and compression. Ensures that the necessary directory structure exists before saving
    the file.

    Args:
        filename: The name of the Parquet file to create and save within the "results/all"
            directory.
        results: A list of dictionaries containing the results to be saved as a Parquet file.
    """
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/all"):
        os.makedirs("./results/all")
    filename = "./results/all/"+ filename
    df = pd.DataFrame(results)
    con = duckdb.connect(database=':memory:')
    con.register('df', df)
    con.execute(f"COPY (SELECT * FROM df) TO '{filename}' (FORMAT PARQUET, COMPRESSION ZSTD)")

def merge_parquet_files(folder_path, output_file, batch_size=10):
    """
    Merge all Parquet files in a folder into a single compressed Parquet file.

    Args:
        folder_path (str): Path to the folder containing Parquet files.
        output_file (str): Path to the output compressed Parquet file.
        batch_size (int): Number of files to process in each batch.
    """
    # List all Parquet files in the folder
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

    # Initialize an empty list to store DataFrames
    merged_data = []

    # Process files in batches
    for i in range(0, len(parquet_files), batch_size):
        batch_files = parquet_files[i:i + batch_size]
        batch_data = []

        # Load each file in the batch
        for file in batch_files:
            file_path = os.path.join(folder_path, file)
            df = pd.read_parquet(file_path)
            batch_data.append(df)

        # Concatenate the batch DataFrames
        if batch_data:
            batch_df = pd.concat(batch_data, ignore_index=True)
            merged_data.append(batch_df)

    # Concatenate all batches into a single DataFrame
    if merged_data:
        final_df = pd.concat(merged_data, ignore_index=True)

        # Save the final DataFrame to a compressed Parquet file
        final_df.to_parquet(output_file, compression='snappy')
        print(f"Merged data saved to {output_file}")
    else:
        print("No Parquet files found to merge.")

def concatenate_parquet_files(folder_path, nb_routes, nb_levels, period, c, B):
    # Define the output file name
    output_file = os.path.join(folder_path, f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_b{B}.csv")

    # Define the pattern to match input files
    #input_pattern = f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_i*.parquet"

    # Get a list of all files matching the pattern
    input_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith(f"results_r{nb_routes}_l{nb_levels}_p{period}_c{c}_b{B}_i") and f.endswith(".parquet")]

    if not input_files:
        print("No files matching the pattern were found.")
        return

    # Use DuckDB to concatenate the files
    con = duckdb.connect(database=':memory:')

    # Create a table from the first file to get the schema
    con.execute(f"CREATE TABLE results AS SELECT * FROM parquet_scan('{input_files[0]}')")

    # Insert data from the remaining files
    for file in input_files[1:]:
        con.execute(f"INSERT INTO results SELECT * FROM parquet_scan('{file}')")

    # Write the concatenated result to a new Parquet file
    con.execute(f"COPY results TO '{output_file}' (FORMAT PARQUET, COMPRESSION 'SNAPPY')")

    print(f"Concatenated file saved as {output_file}")

    for file in input_files:
        os.remove(file)
        print(f" {file} removed.")

# ----------- MULTIPROCESSED FUNCTIONS ------------

import concurrent.futures
import multiprocessing

def worker_compare_all(nb_routes, nb_levels, period, c, B, inst_id, folder_path):
    """
    Compares multiple metaheuristics for a given network configuration and prints
    the start and end of the process for the specified parameters. This function
    retrieves dataset instances, initializes the necessary parameters, runs a
    network simulation with margin extraction, and compares different metaheuristic
    approaches.

    Args:
        nb_routes (int): The number of routes in the network.
        nb_levels (int): The number of levels in the network hierarchy.
        period (int): The period over which the network operates.
        c (float): A numerical parameter used for network configuration.
        inst_id (int): The unique identifier of the specific dataset instance to be used.
    """
    print(f"---- START {nb_routes}-{nb_levels}-{period}-{c}-{inst_id} ----")
    data = get_instance_from_dataset(nb_routes, nb_levels, period, c, inst_id)
    instance, S, Fs, weight_matrix = data[4], data[5], data[6], data[7]
    mrd_initial, supervisor_matrix, _ = run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, int(c), B)
    compare_metaheuristics(instance, Fs, weight_matrix, nb_routes, nb_levels, period, c, B, mrd_initial, supervisor_matrix, S, inst_id, folder_path)
    print(f"---- END {nb_routes}-{nb_levels}-{period}-{c}-{inst_id} ----")

def compare_all_multiprocessed(r_routes, r_levels, r_period, r_c, r_B, r_instances, folder_path = "./results/all"):
    """
    Compares and processes multiple configurations in parallel using multiprocessing, aggregating
    data results into combined files after all individual comparisons are completed.

    This function executes comparisons in a multiprocessing environment by evaluating combinations
    of routes, levels, periods, instances, and coefficients. After completing these tasks, it
    merges the outputs into consolidated Parquet files for each configuration.

    Args:
        r_routes (list[int]): A list containing the numbers of routes to consider in the configurations.
        r_levels (list[int]): A list containing the numbers of levels to consider in the configurations.
        r_period (list[int]): A list containing the periods to evaluate.
        r_c (list[int]): A list of coefficients to adjust processing or evaluation constraints.
        r_instances (range): A list of unique instance identifiers for individual runs.
        folder_path (str, optional): Directory path where results will be stored. Defaults to "./results/all".
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./results/all"):
        os.makedirs("./results/all")
    if not os.path.exists(f"./results/all/{timestamp}"):
        os.makedirs(f"./results/all/{timestamp}")

    folder_path = f"./results/all/{timestamp}/"
    # Get the number of available CPU cores and use half of them
    # num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = max(1, multiprocessing.cpu_count())
    #num_workers = 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for nb_routes in r_routes:
            for nb_levels in r_levels:
                for period in r_period:
                    for c in r_c:
                        for B in r_B:
                            for inst_id in r_instances:
                                futures.append(executor.submit(worker_compare_all, nb_routes, nb_levels, period, int(c),B, inst_id, folder_path))

        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    # Merge parquet files after all comparisons are done
    for nb_routes in r_routes:
        for nb_levels in r_levels:
            for period in r_period:
                for c in r_c:
                    for B in r_B:
                        # concatenate_parquet_files(folder_path, nb_routes, nb_levels, period, int(c), B)
                        concatenate_csv_files(folder_path, nb_routes, nb_levels, period, int(c), B)
                    # merge_parquet_files("./results/all", f"./results/results_r{nb_routes}_l{nb_levels}_p{period}_c{c}.parquet", 10)

# ----------- NOT USED ANYMORE ------------

def compare_noise(nb_routes, nb_levels, period, c, temperature_threshold=0.01, initial_temperature=1.0, cooling_rate=0.95, noise=0.1):
    instance, Fs, weight_matrix, B = generate_instance_and_traffic(nb_routes, nb_levels, period, c)
    mrd_initial, supervisor_matrix = run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, c, B)

    S = noise
    results = [[],[],[],[],[],[]]

    while S < 1:
        start_time = time.time()
        best_supervisor, mrd_recuit, temperature_de_convergence = recuit_simule(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix, temperature_threshold, initial_temperature, cooling_rate, S)
        computation_time = time.time() - start_time

        results[1].append(S)
        results[2].append(mrd_initial)
        results[3].append(mrd_recuit)
        results[4].append(computation_time)
        results[5].append(temperature_de_convergence)
        S+=0.1
    return results

def compare_perturbation_modes(nb_routes, nb_levels, period, c, temperature_threshold=0.01, initial_temperature=1.0, cooling_rate=0.95, noise=0.1):
    instance, Fs, weight_matrix, B = generate_instance_and_traffic(nb_routes, nb_levels, period, c)
    mrd_initial, supervisor_matrix = run_one_network_with_margin_extraction(instance, Fs, weight_matrix, nb_routes, period, c, B)

    results = [[],[],[],[],[],[]]

    modes = ["fixed","random","nr_fixed","nr_random"]

    for m in modes:
        start_time = time.time()
        best_supervisor, mrd_recuit, temperature_de_convergence = recuit_simule(instance, Fs, weight_matrix, nb_routes, period, c, B, supervisor_matrix, temperature_threshold, initial_temperature, cooling_rate, noise, num_rows_to_perturb=m)
        computation_time = time.time() - start_time

        results[1].append(m)
        results[2].append(mrd_initial)
        results[3].append(mrd_recuit)
        results[4].append(computation_time)
        results[5].append(temperature_de_convergence)
    return results

def start_and_compare_multiple_run(nb_routes, nb_levels, period, c, parameter_to_compare, i_max=100):
    final_results = []
    i = 0

    if parameter_to_compare == "strength":
        while i < i_max:
            print("========= Itération n° ", i, " =========")
            results = compare_noise(nb_routes, nb_levels, period, c)
            for _ in range(len(results[1])):
                results[0].append(i)
            final_results.append(results)
            i += 1
    elif parameter_to_compare == "modes":
        while i < i_max:
            print("========= Itération n° ", i, " =========")
            results = compare_perturbation_modes(nb_routes, nb_levels, period, c)
            for _ in range(len(results[1])):
                results[0].append(i)
            final_results.append(results)
            i += 1

    print("final results :", final_results)

    outputname = f"./results/{parameter_to_compare}/output_" + time.strftime("%Y%m%d-%H%M%S") + ".csv"

    with open(outputname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(["id", {parameter_to_compare}, "MRD Initial", "MRD Final", "Computation Time", "Temperature"])
        for dataset in final_results:
            for row in zip(*dataset):
                writer.writerow(row)

def print_result_matrix(result_matrix):
    headers = ["Algorithm", "Initial MRD", "Final MRD", "Computation Time (s)", "Temperature", "Best Supervisor"]
    print("=== Résultats ===")
    print("{:<20} {:<15} {:<15} {:<25} {:<15} {:<15}".format(*headers))
    for row in result_matrix:
        row[5] = str(row[5])
        print("{:<20} {:<15} {:<15} {:<25} {:<15} {:<15}".format(*row))

def concatenate_noise_results(output_filename='combined_results.csv'):
    """
    Concatenate all CSV files in the results/noise directory into one file, adding an ID column.
    After concatenation, deletes the original files.

    Args:
        output_filename (str): Name of the output combined CSV file
    """

    output_filename = time.strftime("%Y%m%d-%H%M%S") +"-"+ output_filename

    noise_dir = Path('results/noise')
    output_path = noise_dir / output_filename

    # List all CSV files
    csv_files = list(noise_dir.glob('tc_*.csv'))

    if not csv_files:
        print("No CSV files found in results/noise directory")
        return

    # Read and combine all CSV files with an ID
    dfs = []
    for idx, file in enumerate(csv_files, 1):
        df = pd.read_csv(file)
        df['id'] = idx
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save combined results
    combined_df.to_csv(output_path, index=False)

    # Delete original files
    for file in csv_files:
        os.remove(file)

    print(f"Combined {len(csv_files)} files into {output_filename}")
    print(f"Added IDs from 1 to {len(csv_files)}")
