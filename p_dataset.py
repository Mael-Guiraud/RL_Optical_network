import os
import random
import duckdb
import multiprocessing
import pandas as pd
import numpy as np
from multiprocessing import Value, Lock
from p_supervisor import generate_instance_and_traffic_bis

# Variables for multiprocessing
counter = Value('i', 0)
lock = Lock()
OUTPUT_DIR = "instances_parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== Dataset generation ==========
def generate_and_store_instance(i, nb_routes, nb_levels, period, c):
    """
    Generates multiple instances of specified configurations and stores them as a compressed
    Parquet file in the defined output directory.

    This function creates and stores a dataset of generated instances, where each row represents
    an instance defined by the configuration parameters provided. The generated data is stored in
    a compressed Parquet file format, utilizing efficient compression and storage.

    Args:
        i (int): Number of instances to generate. Must be greater than 0.
        nb_routes (int): Number of routes for the instances.
        nb_levels (int): Number of levels for the instances.
        period (int): Time period for the instances.
        c (float): Capacity or specific parameter value for the instances.

    """
    # Check if the number of instances to generate is valid
    if i <= 0:
        print("Error, i must be greater than 0.")
        return

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=['nb_routes', 'nb_levels', 'period', 'c', 'instance', 'S', 'Fs', 'weight_matrix', 'B'])
    for _ in range(i):
        instance, S, Fs, weight_matrix, B = generate_instance_and_traffic_bis(nb_routes, nb_levels, period, c)
        new_row = pd.DataFrame([{
            "nb_routes": nb_routes,
            "nb_levels": nb_levels,
            "period": period,
            "c": c,
            "instance": repr(instance),
            "S": repr(S),
            "Fs": repr(Fs),
            "weight_matrix": repr(weight_matrix),
            "B": B
        }])
        df = pd.concat([df, new_row], ignore_index=True)

    # Create the output directory if it doesn't exist
    filename = f"r{nb_routes}_l{nb_levels}_p{period}_c{int(c)}.parquet"
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Copy df to a compressed Parquet file
    con = duckdb.connect()
    con.register('df', df)
    con.execute(f"COPY df TO '{filepath}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()

def worker(task_args):
    i, nb_routes, nb_levels, period, c = task_args
    try:
        generate_and_store_instance(i, nb_routes, nb_levels, period, c)
        with lock:
            counter.value += 1
            print(f"✓ Instance {counter.value} — ({nb_routes}, {nb_levels}, {period}, {c})")
    except Exception as e:
        print(f"⚠️ Erreur: {nb_routes}, {nb_levels}, {period}, {c} → {e}")

def generate_dataset(i):
    """
    Generates a dataset by creating task instances with various combinations of
    parameters, and processes these tasks using multiprocessing.

    Args:
        i: An identifier for the dataset being generated. Used to distinguish tasks.
    """
    # r_routes = range(5, 8+1)
    r_routes = range(5, 21)
    r_levels = range(2, 5+1)
    r_period = range(10, 100+1, 10)

    tasks = []
    for nb_routes in r_routes:
        for nb_levels in r_levels:
            for period in r_period:
                for c in [1, 2, 3, period / 2]:
                    tasks.append((i, nb_routes, nb_levels, period, c))

    print(f"▶️ Démarrage — {len(tasks)} instances à générer pour i={i}")

    cpu = round(multiprocessing.cpu_count()/2)
    with multiprocessing.Pool(processes=cpu) as pool:
        pool.map(worker, tasks)

# ========= Database functions ==========

def merge_parquet_files(output_file="dataset.parquet"):
    """
    Merge all Parquet files in the OUTPUT_DIR into a single Parquet file.
    """
    con = duckdb.connect()
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{OUTPUT_DIR}/*.parquet')
        ) TO '{output_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()

def check_dataset_files():
    """
    Check if all expected Parquet files exist in the OUTPUT_DIR based on the parameter ranges
    used in generate_dataset.

    Returns:
        tuple: (bool, list) - (True if all files exist, list of missing files)
    """
    r_routes = range(5, 8 + 1)
    r_levels = range(2, 5 + 1)
    r_period = range(10, 100 + 1, 10)

    missing_files = []
    all_present = True

    for nb_routes in r_routes:
        for nb_levels in r_levels:
            for period in r_period:
                for c in [1, 2, 3, period / 2]:
                    filename = f"r{nb_routes}_l{nb_levels}_p{period}_c{int(c)}.parquet"
                    filepath = os.path.join(OUTPUT_DIR, filename)

                    if not os.path.exists(filepath):
                        missing_files.append(filename)
                        all_present = False

    if all_present:
        print("✓ All expected files are present in the instances_parquet folder")
    else:
        print(f"⚠️ Missing {len(missing_files)} files:")
        for file in missing_files:
            print(f"  - {file}")

    return all_present, missing_files

# ========== Data conversion ==========

def deserialize(data):
    """
    Automatically convert string representation of data to its original type.
    Detects regular lists vs numpy array structures (Fs/S) based on string content.
    """
    if isinstance(data, (int, float,np.integer)):
        return data
    if isinstance(data, str):
        if "array" in data:
            # Handle Fs/S data containing numpy arrays
            data = data.replace("array", "np.array")
            return [[[np.array(arr, dtype=np.float64) for arr in sublist]
                     for sublist in inner_list]
                     for inner_list in eval(data, {'np': np})]
        else:
            # Handle regular lists
            #print(data, type(data))
            return eval(data, {'np': np})

# ========== High level functions ==========

def get_instance_from_dataset(nb_routes, nb_levels, period, c, inst_id=None):
    """
    Retrieve an instance from a Parquet file based on the given parameters.

    Args:
        nb_routes (int): The number of routes in the instance.
        nb_levels (int): The number of levels in the instance.
        period (int): The period associated with the instance.
        c (int or float): A parameter used to identify the instance.
        inst_id (int, optional): The index of the instance to retrieve. If None, a random instance is selected.

    Returns:
        tuple: A tuple containing the deserialized data of the selected instance,
               or None if no matching file or instance is found.

    Notes:
        - The function looks for a Parquet file in the `OUTPUT_DIR` directory with a name
          matching the format `r{nb_routes}_l{nb_levels}_p{period}_c{c}.parquet`.
        - If the file exists, it retrieves all rows from the file, selects one instance
          and deserializes its data.
        - If the file does not exist or no matching instance is found, the function returns None.
    """
    # Construct the file name based on the input parameters
    file = f"r{nb_routes}_l{nb_levels}_p{period}_c{c}.parquet"

    # Check if the file exists in the output directory
    if not os.path.exists(os.path.join(OUTPUT_DIR, file)):
        print(f"File {file} does not exist.")
        return None

    # Connect to the DuckDB database and read the Parquet file
    con = duckdb.connect()
    con.execute(f"""
        SELECT * FROM read_parquet('{OUTPUT_DIR}/{file}') 
        WHERE nb_routes = {nb_routes} AND nb_levels = {nb_levels} AND period = {period} AND c = {c}
    """)
    results = con.fetchall()
    con.close()

    # If results are found, select one instance
    if results and inst_id is None:
        # If i is None, select a random instance
        selected_instance = random.choice(results)
    elif results and inst_id < len(results):
        # If i is provided, select the instance at index i
        selected_instance = results[inst_id]
    else:
        # Print a message if no matching instance is found
        print(f"No matching instance found for parameters: nb_routes={nb_routes}, nb_levels={nb_levels}, period={period}, c={c}, id={inst_id}")
        return None
    return tuple(deserialize(data) for data in selected_instance)
import lb_traffic
# ========== Main function ==========
if __name__ == "__main__":
    #multiprocessing.freeze_support()
    #generate_dataset(10)
    #print(f"🎉 Terminé. Total généré : {counter.value}")
    #duckdb.sql("SELECT count(*) FROM 'instances_parquet/*.parquet'").show()
    instance = get_instance_from_dataset(5, 2, 10, 2)
    print(f"Instance: {lb_traffic.count_ones_in_most_loaded_matrix(instance[6])}")
    #check_dataset_files()