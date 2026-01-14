#!/usr/bin/env python3

import lb_topo
import lb_traffic
from algo import run_test_case
import numpy as np
import pulp
import sys
from pathlib import Path
import csv
import time
import traceback 
from greedy import min_avg_delay,mrd_buffer,average_delay_buffer
# Importer le script ou appeler une fonction
from Scriptv2 import run_LRI


def reset_schedule(f):
    for i,level in enumerate(f):
        if i>0:
            for j,router in enumerate(level):
                for k,cycle in enumerate(router):
                    for l,route in enumerate(cycle):
                        for m,tic in enumerate(route):
                            f[i][j][k][l][m] =0
def generate_instance_and_traffic(nb_routes,nb_levels,period,c):
    instance= None
    while not instance:
        instance,nb_fails=lb_topo.generate_instance(nb_routes,nb_levels)
    if instance:
        #print(lb_topo.check_weakly_coherent(instance,nb_routes),lb_topo.check_strongly_coherent(instance,nb_routes))
        Fs,weight_matrix,_,_,_,B = lb_traffic.generate_random_traffic_no_save(instance,period,nb_routes,c)
    else:
        print("Error, no instance found")
        exit(1)
    return instance,Fs,weight_matrix,B
def run_one_network(instance,Fs,weight_matrix,nb_routes,period,c,B,mode):
    taumin = []
    for rid in range(nb_routes):
        taumin.append(lb_traffic.route_length(rid,weight_matrix,instance))
    F = Fs[0]
    reset_schedule(Fs)
    S = lb_traffic.init_all_node_matrix(instance,period,nb_routes)
    delays = lb_traffic.init_all_node_matrix(instance,period,nb_routes)
    for i,level in enumerate(Fs):
        if i>0:
            for j,router in enumerate(level):
                for k,cycle in enumerate(router):
                    if mode == "POLY" or mode == "POLYOPTI":
                        
                        found = False
                        margin = 0
                        while not found and margin < period:
                            tasks = []
                            #print(i,j,k)
                            for l in instance[i][j][k]:
                                previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance,i-1,l)
                                #### FIRST STEP, TRANSFORM INTO THE EXPECTED FORMAT TO SOLVE THE ONE NODE PROBLEM
                                
                                for t in range(period):
                                    if Fs[i-1][previous_r][previous_cycle][l][t] == 1:
                                        r_i = (t + weight_matrix[i-1][previous_r][previous_cycle]) % period
                                        d= delays[i-1][previous_r][previous_cycle][l][t]
                                        d_i = r_i+margin
                                        tasks.append({'id': f'task_{l}_{r_i}', 'r_i': r_i, 'd_i': d_i, 'delay':d,'tau':taumin[l]})
                            #print(tasks)
                            if len(tasks) == 0:
                                print("Error, no task found")
                                #print(Fs[i-1])
                                exit(1)
                            if len(tasks) == 1:
                                parts = tasks[0]['id'].split('_')
                                route_id = int(parts[1])  # Le premier numéro après 'task'
                                time = int(parts[2])   # Le second numéro
                                Fs[i][j][k][route_id][time] = 1
                                found = True
                            else:
                                if mode == "POLYOPTI":
                                    result = run_test_case(nb_routes,tasks,period,c,B,pulp.PULP_CBC_CMD(msg=False),"opti")
                                else:
                                    result = run_test_case(nb_routes,tasks,period,c,B,pulp.PULP_CBC_CMD(msg=False),"basic")
                                if len( result["assigned_dates"]) != len(tasks):
                                    margin += 1
                                    found = False
                                  
                                else:
                                    #### SECOND STEP, FILL S
                                    for key, value in result["assigned_dates"].items():
                           
                                        # Extraction des numéros après "task"
                                        parts = key.split('_')
                                        route_id = int(parts[1])  # Le premier numéro après 'task'
                                        time = int(parts[2])   # Le second numéro
                                        new_time = value       # La valeur associée
                                        delai =(new_time - time) % period if (((new_time - time) % period) >= c)or(((new_time - time) % period) == 0) else (new_time - time) % period + period
                                        previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance,i-1,route_id)
                                        Fs[i][j][k][route_id][new_time] = 1

                                        delays[i][j][k][route_id][new_time] = delays[i-1][previous_r][previous_cycle][route_id][(time-weight_matrix[i-1][previous_r][previous_cycle])%period] + delai
                                        
                                        #print("Ajout de delai ",i,j,k,route_id,new_time,delai,time, previous_cycle,previous_r,delays[i-1][previous_r][previous_cycle][route_id][time])
                                        S[i][j][k][route_id][time] = delai
                                        #print("Route",route_id,"from",time,"to",new_time,"with delay",(new_time - time) % period if ((new_time - time) % period) >= c else (new_time - time) % period + period)
                                    found = (not result["error"] and result["status"] != "Infeasible") 
                                    print(result["error_details"],result["status"])
                                    margin += 1
                        if not found:
                            return -1,None,None
                    elif mode == "LRI":
                        F_matrix = np.zeros((nb_routes, period), dtype=int)
                        delay = np.zeros((nb_routes, period), dtype=int)
                        #For each route of the router
                        for route_id in instance[i][j][k]:
                            #find the previous router and cycle
                            previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance,i-1,route_id)
                            weight=weight_matrix[i-1][previous_r][previous_cycle]
                            for t in range(period):
                                if Fs[i-1][previous_r][previous_cycle][route_id][t] == 1:
                                    F_matrix[route_id][(t+weight)%period] = 1
                                    delay[route_id][(t+weight)%period] = delays[i-1][previous_r][previous_cycle][route_id][t] 
                        _, _, _, _, _, _, _, _, _, _, _, _, best_S = run_LRI(period, nb_routes, F_matrix, taumin, delay, c, B, 0.01, 20000)
                        for l,route_id in enumerate(cycle):
                            for t in range(period):
                                if F_matrix[l][t] == 1:
                                    Fs[i][j][k][l][(t+best_S[l][t])%period] = 1
                                    S[i][j][k][l][t] = best_S[l][t]
                                    delays[i][j][k][l][(t+best_S[l][t])%period] = best_S[l][t]+delay[l][t]
                    elif mode =="Greedy":
                        arrivals = []                        
                        for tic in range(period):
                            arrivals.append([])
                            for rou in range(len(cycle)):
                                previous_r, previous_cycle = lb_traffic.find_id_router_and_cycle(instance,i-1,rou)
                                if Fs[i-1][previous_r][previous_cycle][rou][tic] == 1:   
                                    r_i = (tic + weight_matrix[i-1][previous_r][previous_cycle]) % period
                                    d= delays[i-1][previous_r][previous_cycle][rou][tic]
                                    arrivals[tic].append([(0, taumin[rou], d), [rou, r_i]])
                        res = min_avg_delay(arrivals, c, B, mrd_buffer)
                        if res== -1:
                            print("Error, no solution found for Greedy")
                            return -1,None,None
                        for elem in res[2]:
                            route_id = elem[1][0]
                            arrivals_time = elem[1][1]
                            add_delay = elem[0]
                            Fs[i][j][k][route_id][(arrivals_time+add_delay)%period] = 1
                            S[i][j][k][route_id][arrivals_time] = add_delay
                            delays[i][j][k][route_id][(arrivals_time+add_delay)%period] = add_delay + delays[i-1][previous_r][previous_cycle][route_id][(arrivals_time-weight_matrix[i-1][previous_r][previous_cycle])%period]
    
    #print("Pour poly :")
    #print("F",Fs)
    #print("S",S)
    #print("Delays",delays)
    #print("Fs",Fs)

    erreurs = False
    #Vérification des Fs et des délais
    if not lb_traffic.check_FS(Fs):
        print("Error 1 ")
        erreurs = True
    if not lb_traffic.check_FS2(Fs):
        print("Error 2")
        erreurs = True
    if not lb_traffic.check_FS3(Fs,nb_routes):
        print("Error 3",lb_traffic.check_FS3_message(Fs,nb_routes))
        erreurs = True
    if not lb_traffic.check_delays(delays,c):
        print("Error 4")
        erreurs = True
    if not lb_traffic.check_matrices_Fs_delays(Fs,delays):
        print("Error 5")
        erreurs = True
    if not lb_traffic.check_if_pattern_ok(S,Fs,weight_matrix,period,instance):
        print("Error 6")
        erreurs = True
    if not lb_traffic.check_buffer_min(S,c):
        print("Error 7")
        erreurs = True
    if not lb_traffic.check_buffer_max(S,Fs,weight_matrix,period,instance,B):
        print("Error 8")
        erreurs = True
    if erreurs:
        print("Mode",mode)
        print("F:",F)
        print("instance:",instance)
        print("weight_matrix:",weight_matrix)
        print("B",B)
        print("FS",Fs)
        print("Delays",delays)
        print("S",S)

    """
    #find max delay in delays
    max_delay = 0
    route_max_delay = -1
    for j,router in enumerate(delays[-1]):
        for k,cycle in enumerate(router):
            for l,route in enumerate(cycle):
                for m,tic in enumerate(route):
                    if delays[-1][j][k][l][m] > max_delay:
                        max_delay = delays[-1][j][k][l][m]
                        route_max_delay = l
    print("Max delay:",max_delay," on route ",route_max_delay)
    print("tau_min:",taumin)    """
    return lb_traffic.max_ratio_delay(delays,weight_matrix,instance),S,delays
    
    
import csv
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
# IMPORTANT: ces fonctions doivent être définies au niveau module (pas imbriquées)
# def generate_instance_and_traffic(...)
# def run_one_network(...)


def one_test(test_id, nb_routes, nb_levels, period, c,B):
    """
    Exécute 1 test complet et renvoie un dict de résultats.
    """
    try:
        from p_dataset import get_instance_from_dataset
        data = get_instance_from_dataset(nb_routes, nb_levels, period, c, test_id)
        instance, _, Fs, weight_matrix = data[4], data[5], data[6], data[7]
   
        t0 = time.time()
        poly_value = run_one_network(instance, Fs, weight_matrix, nb_routes, period, c, B, "POLY")
        poly_time = time.time() - t0

        t0 = time.time()
        poly_value_opti = run_one_network(instance, Fs, weight_matrix, nb_routes, period, c, B, "POLYOPTI")
        poly_opti_time = time.time() - t0

        t0 = time.time()
        lri_value = run_one_network(instance, Fs, weight_matrix, nb_routes, period, c, B, "LRI")
        lri_time = time.time() - t0

        return {
            "Test": test_id,
            "POLY": poly_value,
            "POLY_time": poly_time,
            "POLYOPTI": poly_value_opti,
            "POLYOPTI_time": poly_opti_time,
            "LRI": lri_value,
            "LRI_time": lri_time,
            "error": None,
        }

    except Exception as e:
        return {
            "Test": test_id,
            "POLY": None,
            "POLY_time": None,
            "POLYOPTI": None,
            "POLYOPTI_time": None,
            "LRI": None,
            "LRI_time": None,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    # Paramètres
    nb_routes = 5
    nb_levels = 2
    period = 10
    c = 2
    num_tests = 100
    B = 3
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_file = "network_results+{}.csv".format(timestamp)

    results = []

    # nb_workers: mets None pour "autant que de coeurs"
    # ou un entier (ex: 4, 8...). Souvent: os.cpu_count() - 1
    nb_workers = os.cpu_count() - 1

    with ProcessPoolExecutor(max_workers=nb_workers) as ex:
        futures = [
            ex.submit(one_test, test_id, nb_routes, nb_levels, period, c,B)
            for test_id in range(1, num_tests + 1)
        ]

        for fut in as_completed(futures):
            r = fut.result()
            if r.get("error"):
                print(f"Test {r['Test']} ERROR: {r['error']}")
                print(r.get("traceback", ""))
                # si tu veux stopper tout le monde au premier échec:
                # for f in futures: f.cancel()
                # break
            else:
                print(
                    f"Test {r['Test']} OK: POLY={r['POLY']}, POLYOPTI={r['POLYOPTI']}, LRI={r['LRI']}"
                )
            results.append(r)

    # Tri par numéro de test (as_completed rend dans un ordre quelconque)
    results.sort(key=lambda d: d["Test"])

    # Écriture CSV
    fieldnames = ["Test", "POLY", "POLY_time", "POLYOPTI", "POLYOPTI_time", "LRI", "LRI_time", "error"]
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    print(f"Results written to {output_file}")

    # Moyennes (sur les tests sans erreur)
    ok = [r for r in results if not r.get("error")]
    if ok:
        avg_poly = sum(r["POLY"] for r in ok) / len(ok)
        avg_poly_opti = sum(r["POLYOPTI"] for r in ok) / len(ok)
        avg_lri = sum(r["LRI"] for r in ok) / len(ok)
        print(f"Average POLY: {avg_poly}")
        print(f"Average POLYOPTI: {avg_poly_opti}")
        print(f"Average LRI: {avg_lri}")
