#!/usr/bin/env python3
import pulp
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def check_task_assignment(task_id, date, task, P, c):
    # [Le code de cette fonction reste inchangé]
    r_i_mod = task['r_i'] % P
    d_i_mod = task['d_i'] % P
    t = date % P
    c_mod = c % P
    r_i_plus_c_mod = (task['r_i'] + c) % P
    available = True

    if not available:
        error_detected = True
        error_details = f"Violation : La tâche {task_id} est assignée à {t}, hors de sa fenêtre de disponibilité.\n"
        return error_detected, error_details

    # Si aucune violation n'est détectée
    error_detected = False
    error_details = ''
    return error_detected, error_details


def is_allowed(t, task, P, c):
    r_i_mod = task['r_i'] % P
    d_i_mod = task['d_i'] % P
    d_i = task['d_i']
    c_mod = c % P
    r_i_plus_c_mod = (task['r_i'] + c) % P
    # Si d_i >= ri + P+c, alors la tâche est disponible à tout moment
    if d_i >= task['r_i'] + P + c:
        return True

    # Exclure les instants entre r_i_mod et r_i_plus_c_mod
    if r_i_mod < r_i_plus_c_mod:
        if (r_i_mod < t < r_i_plus_c_mod) and (d_i_mod < t):
            return False
    elif r_i_mod > r_i_plus_c_mod:
        if (t > r_i_mod or t < r_i_plus_c_mod) and (d_i_mod < t):
            return False
    else:
        # r_i_mod == r_i_plus_c_mod
        if c_mod != 0 and t != r_i_mod:
            return False

    # Déterminer la fenêtre de disponibilité
    if r_i_mod == d_i_mod:
        if task['r_i'] != task['d_i']:
            # La tâche est disponible à tout moment sauf l'intervalle interdit
            return True
        else:
            # La tâche n'est disponible qu'à r_i_mod
            return t == r_i_mod
    elif r_i_mod < d_i_mod:
        return r_i_mod <= t <= d_i_mod
    else:
        return t >= r_i_mod or t <= d_i_mod


def count_for_B0(a, b, c, P):
    """
    Parameters
    ----------
    a : int   # chosen date  (e[1])     modulo P
    b : int   # release time (task r_i) modulo P
    c : int   # conversion time
    P : int   # period

    Returns
    -------
    bool  – True  ↔  edge e contributes to B0
    """

    if a == b:
        return False
    if a < b:

        return True  # case 1
    else:
        e = (b + c) % P  # threshold
        if e > b:  # case 2

            return a < e
        else:  # case 3 (wrap-around)

            return (a > b) or (a < e)


def run_test_case(N, tasks, P, c_value, B_value, solver=pulp.GUROBI(msg=False), mode="opti"):
    total_start_time = time.time()  # Début du chronométrage total
    lp_creation_start_time = time.time()  # Début du chronométrage de la création du LP
    max_additional_delay = (P + c_value - 1) * N
    c = c_value
    B = B_value

    # Les sommets de U sont les tâches
    U = [task['id'] for task in tasks]

    # Les sommets de V sont les dates entre 0 et P - 1
    V = list(range(0, P))

    # Génération des arêtes selon les spécifications corrigées
    E = []

    for task in tasks:
        dates = set()
        dates.add(task['r_i'])
        delai = c
        # new dictionsnary to store the induced delay of each date stored in dates
        dict = {}
        dict2 = {}
        dict[task['r_i']] = 0
        dict2[task['r_i']] = 0
        for t in range(task['r_i'] + c, task['d_i'] + 1):
            dates.add(t % P)
            if t % P != task['r_i']:
                dict[t % P] = -delai
                dict2[t % P] = 0
                if t > task['r_i'] + P:
                    dict2[t % P] = 1
            delai = delai + 1
        # Ajout des arêtes
        for v in dates:
            E.append((task['id'], v, dict[v], dict2[v]))
    task_release_time_mod = {}
    for task in tasks:
        task_release_time_mod[task['id']] = task['r_i'] % P
    # Création du problème de programmation linéaire
    prob = pulp.LpProblem("CouplageMaximal", pulp.LpMaximize)

    # Variables de décision : x_e pour chaque arête e ∈ E
    # Variables continues entre 0 et 1
    x = pulp.LpVariable.dicts('x', E, 0, 1, cat='Continuous')
    # Ajout de la contrainte pour B0
    B0 = pulp.LpVariable('B0', lowBound=0)
    # Fonction objectif : maximiser la somme des variables de décision
    if mode == "opti":
        prob += pulp.lpSum([x[e] * (1 + e[2] / max_additional_delay) for e in E]), "Objectif"
    else:
        prob += pulp.lpSum([x[e] for e in E]), "Objectif"

    # Contraintes pour les tâches (chaque tâche peut être assignée à au plus une date)
    for u in U:
        prob += pulp.lpSum([x[e] for e in E if e[0] == u]) <= 1, f"Tache_{u}"

    # Contraintes pour les dates (chaque date peut avoir au plus une tâche)
    for v in V:
        prob += pulp.lpSum([x[e] for e in E if e[1] == v]) <= 1, f"Date_{v}"

    # Calcul de D(t) comme le nombre de tâches arrivant au buffer à t
    D_t_list = {}
    for t in V:
        D_t_list[t] = len([task for task in tasks if task['r_i'] % P <= t]) + pulp.lpSum([x[e] for e in E if e[3]])

    # Calcul des départs du buffer à chaque instant t
    S_t_list = {}
    for t in V:
        S_t = pulp.lpSum([x[e] for e in E if (e[1] % P) <= t])
        S_t_list[t] = S_t
    task_release_time_mod = {task['id']: task['r_i'] % P for task in tasks}

    prob += B0 == pulp.lpSum(
        x[e] for e in E
        if count_for_B0(a=e[1],
                        b=task_release_time_mod[e[0]],
                        c=c_value,
                        P=P)
    ), "B0"

    for t in V:
        prob += B0 + D_t_list[t] - S_t_list[t] <= B, f"Buffer_Level_{t}"

    lp_creation_end_time = time.time()  # Fin du chronométrage de la création du LP
    lp_creation_time = lp_creation_end_time - lp_creation_start_time

    # Résolution du problème avec le solveur spécifié
    solver_start_time = time.time()  # Début du chronométrage du solveur
    status = prob.solve(solver)
    solver_end_time = time.time()  # Fin du chronométrage du solveur
    solver_time = solver_end_time - solver_start_time

    total_end_time = time.time()  # Fin du chronométrage total
    total_time = total_end_time - total_start_time

    # [Le reste du code de la fonction reste inchangé]
    # Préparation des informations pour le rapport d'erreur
    error_details = ''
    error_detected = False

    if pulp.LpStatus[status] == 'Optimal':
        # Initialisation
        assigned_tasks = {}
        delays = {}
        assigned_dates = {}
        integer_solution = True
        epsilon = 1e-5  # Tolérance pour la vérification des entiers
        B0_value = pulp.value(B0)
        # print("valeur de la solution",prob.objective.value())
        # print("valeur de B0",B0_value)
        # print("Tasks:",tasks)
        # print("valeur des variables")
        # for e in E:
        #    print(e,pulp.value(x[e]))
        # Collecte des tâches assignées et vérification des variables entières
        for e in E:
            val = pulp.value(x[e])

            if val > 0:
                if abs(val - round(val)) > epsilon:
                    integer_solution = False
                    error_detected = True
                    error_details += f"Variable x[{e}] a une valeur non entière : {val}\n"
                else:
                    val = round(val)
                assigned_tasks[e] = val

        if not integer_solution:
            error_detected = True
            error_details += "La solution contient des variables non entières.\n"
        else:
            # Vérification des contraintes de release time et deadline
            for (task_id, date, delai, _), val in assigned_tasks.items():
                # Récupération des informations de la tâche
                task = next(task for task in tasks if task['id'] == task_id)
                # Appel de la fonction de vérification
                violation, details = check_task_assignment(task_id, date, task, P, c)
                if violation:
                    error_detected = True
                    error_details += details

                    # Vérifier que le buffer ne dépasse jamais B
            if not error_detected:
                # 1) arrivées ponctuelles A(t)
                A_t = {t: 0 for t in V}
                for task in tasks:
                    A_t[task['r_i'] % P] += 1

                # 2) départs ponctuels S(t) (tirés de la solution)
                S_t = {t: 0 for t in V}
                for (task_id, date, _, _), val in assigned_tasks.items():
                    if val == 1:
                        S_t[date] += 1

                # 3) parcours cumulatif
                cumul_arr, cumul_dep = 0, 0
                for t in V:
                    cumul_arr += A_t[t]
                    cumul_dep += S_t[t]
                    buffer_level = B0_value + cumul_arr - cumul_dep
                    if buffer_level > B + epsilon:
                        error_detected = True
                        error_details += (
                            f"Violation : buffer {buffer_level} > B={B} à t={t} "
                            f"(B0={B0_value}, cum_arr={cumul_arr}, cum_dep={cumul_dep})\n"
                        )
                        break

            for (task_id, date, delai, _), val in assigned_tasks.items():
                if val == 1:
                    assigned_dates[task_id] = date
                    task = next(task for task in tasks if task['id'] == task_id)
                    r_i = task['r_i']
                    delay = (date - r_i) % P if (((date - r_i) % P) >= c) or (((date - r_i) % P) == 0) else (date - r_i) % P + P
                    delays[task_id] = delay
            if len(assigned_dates) != len(tasks):
                error_detected = True
                error_details += "Certaines tâches n'ont pas été assignées à une date.\n"

    else:
        # Si le problème n'est pas résolu de manière optimale, nous ne considérons pas cela comme une erreur
        error_detected = True
        error_details += f"Aucune solution optimale trouvée pour cette instance. Statut : {pulp.LpStatus[status]}\n"

        delays = {}
        assigned_dates = {}

    # Retourner le résultat avec les temps mesurés

    return {
        'params': {'N': N, 'P': P, 'c': c_value, 'B': B_value},
        'error': error_detected,
        'tasks': tasks,
        'error_details': error_details,
        'status': pulp.LpStatus[status],
        'total_time': total_time,
        'lp_creation_time': lp_creation_time,
        'solver_time': solver_time,
        'delays': delays,
        'assigned_dates': assigned_dates,
        'prob': prob
    }


def exps_time():
    import time
    import matplotlib.pyplot as plt

    # Listes pour stocker les temps d'exécution moyens
    Ps = range(20, 201, 20)  # P et B de 20 à 200 par incréments de 20
    num_instances = 100  # Nombre d'instances à exécuter pour chaque valeur de P

    # Dictionnaires pour stocker les temps pour chaque solveur
    avg_times_default = {'total': [], 'lp_creation': [], 'solver': []}
    avg_times_gurobi = {'total': [], 'lp_creation': [], 'solver': []}

    for P in Ps:
        B = P
        N = int(0.95 * P)
        c = 2  # Vous pouvez ajuster c si nécessaire

        # Listes pour stocker les temps d'exécution pour chaque instance
        times_default = {'total': [], 'lp_creation': [], 'solver': []}
        times_gurobi = {'total': [], 'lp_creation': [], 'solver': []}

        for instance in range(num_instances):
            # Génération des tâches
            tasks = []
            for i in range(N):
                r_i = random.randint(0, P - 1)
                d_i = random.randint(r_i, r_i + P - 1)  # Assurer que r_i ≤ d_i
                tasks.append({'id': f'task_{i}', 'r_i': r_i, 'd_i': d_i})

            # Exécution avec le solveur par défaut de PuLP
            solver_default = pulp.PULP_CBC_CMD(msg=False)
            result_default = run_test_case(N, tasks, P, c, B, solver_default)
            times_default['total'].append(result_default['total_time'])
            times_default['lp_creation'].append(result_default['lp_creation_time'])
            times_default['solver'].append(result_default['solver_time'])

            # Exécution avec Gurobi
            solver_gurobi = pulp.GUROBI(msg=False)
            result_gurobi = run_test_case(N, tasks, P, c, B, solver_gurobi)
            times_gurobi['total'].append(result_gurobi['total_time'])
            times_gurobi['lp_creation'].append(result_gurobi['lp_creation_time'])
            times_gurobi['solver'].append(result_gurobi['solver_time'])

        # Calcul des temps d'exécution moyens pour cette valeur de P
        avg_times_default['total'].append(sum(times_default['total']) / num_instances)
        avg_times_default['lp_creation'].append(sum(times_default['lp_creation']) / num_instances)
        avg_times_default['solver'].append(sum(times_default['solver']) / num_instances)

        avg_times_gurobi['total'].append(sum(times_gurobi['total']) / num_instances)
        avg_times_gurobi['lp_creation'].append(sum(times_gurobi['lp_creation']) / num_instances)
        avg_times_gurobi['solver'].append(sum(times_gurobi['solver']) / num_instances)

        print(f"P = {P}")
        print(f"  Solveur par défaut : Temps total moyen = {avg_times_default['total'][-1]:.4f}s, Temps LP = {avg_times_default['lp_creation'][-1]:.4f}s, Temps solveur = {avg_times_default['solver'][-1]:.4f}s")
        print(f"  Gurobi            : Temps total moyen = {avg_times_gurobi['total'][-1]:.4f}s, Temps LP = {avg_times_gurobi['lp_creation'][-1]:.4f}s, Temps solveur = {avg_times_gurobi['solver'][-1]:.4f}s")

    # Tracer les temps d'exécution moyens pour le solveur par défaut
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(Ps, avg_times_default['total'], marker='o', label='Temps total')
    plt.plot(Ps, avg_times_default['lp_creation'], marker='s', label='Temps création LP')
    plt.plot(Ps, avg_times_default['solver'], marker='^', label='Temps solveur')
    plt.xlabel('Valeur de P (et B)')
    plt.ylabel('Temps moyen (secondes)')
    plt.title('Solveur par défaut de PuLP')
    plt.legend()
    plt.grid(True)

    # Tracer les temps d'exécution moyens pour Gurobi
    plt.subplot(1, 2, 2)
    plt.plot(Ps, avg_times_gurobi['total'], marker='o', label='Temps total')
    plt.plot(Ps, avg_times_gurobi['lp_creation'], marker='s', label='Temps création LP')
    plt.plot(Ps, avg_times_gurobi['solver'], marker='^', label='Temps solveur')
    plt.xlabel('Valeur de P (et B)')
    plt.ylabel('Temps moyen (secondes)')
    plt.title('Solveur Gurobi')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def exp_same_result():
    random.seed(time.time())  # Initialiser le générateur de nombres aléatoires
    P = 3
    B = 5
    N = 3
    c = 3

    tasks = []
    for i in range(N):
        r_i = random.randint(0, P - 1)
        d_i = random.randint(r_i + P, r_i + P + c - 1)  # Assurer que r_i ≤ d_i
        tasks.append({'id': f'task_{i}', 'r_i': r_i, 'd_i': d_i})
    # Exécution avec Gurobi
    tasks = [{'id': 'task_0', 'r_i': 0, 'd_i': 0},
             {'id': 'task_1', 'r_i': 0, 'd_i': 4},
             {'id': 'task_2', 'r_i': 0, 'd_i': 5}]
    result = run_test_case(N, tasks, P, c, B, pulp.PULP_CBC_CMD(msg=False))

    if result['error']:
        print("Erreur détectée :")
        print(result['error_details'])
    else:
        print("Aucune erreur détectée.")
        print(f"Statut : {result['status']}")
        # Afficher les délais pour chaque tâche
        print("Délai ajouté pour chaque tâche :")
        print("Tâche\tRelease Time\tAssigned Date\tDélai")
        for task_id in sorted(result['delays'].keys()):
            task = next(task for task in tasks if task['id'] == task_id)
            r_i = task['r_i']
            assigned_date = result['assigned_dates'][task_id]
            delay = result['delays'][task_id]
            print(f"{task_id}\t{r_i}\t\t{assigned_date}\t\t{delay}")


def exp_varying_deadlines():
    P = 20
    B = P
    N = int(0.95 * P)
    c = 2

    # Générer une instance unique de tâches
    tasks = []
    for i in range(N):
        r_i = random.randint(0, P - 1)
        d_i = random.randint(r_i, r_i + P - 1)  # Assurer que r_i ≤ d_i
        tasks.append({'id': f'task_{i}', 'r_i': r_i, 'd_i': d_i})

    # Dictionnaire pour stocker les délais pour chaque tâche à chaque augmentation de deadline
    delays_per_task = {task['id']: [] for task in tasks}

    # Liste pour les valeurs de l'augmentation des deadlines
    deadline_increments = list(range(0, P + 1))  # De 0 à P inclus

    solver = pulp.GUROBI(msg=False)  # Vous pouvez changer de solveur si nécessaire

    # Pour chaque valeur d'augmentation de la deadline
    for delta in deadline_increments:
        print("delta =", delta)
        # Copier les tâches pour cette itération
        tasks_modified = []
        for task in tasks:
            # Augmenter la deadline de delta
            new_d_i = task['d_i'] + delta
            tasks_modified.append({'id': task['id'], 'r_i': task['r_i'], 'd_i': new_d_i})

        # Résoudre le problème avec les tâches modifiées
        result = run_test_case(N, tasks_modified, P, c, B, solver)

        # Vérifier s'il y a une erreur
        if result['error']:
            print(f"Erreur détectée pour delta = {delta}:")
            print(result['error_details'])
            # Ajouter None pour les tâches non résolues
            for task_id in delays_per_task.keys():
                delays_per_task[task_id].append(None)
        else:
            # Récupérer les délais pour chaque tâche
            for task_id in delays_per_task.keys():
                delay = result['delays'].get(task_id, None)
                delays_per_task[task_id].append(delay)

    # --- Début de la partie modifiée pour la visualisation ---

    # Convertir delays_per_task en DataFrame
    df_delays = pd.DataFrame.from_dict(delays_per_task, orient='index', columns=deadline_increments)

    # Convertir les None en NaN pour la gestion des valeurs manquantes
    df_delays = df_delays.replace({None: np.nan})

    # Optionnel : Réindexer les tâches avec des indices numériques pour la lisibilité
    df_delays.index = range(len(df_delays))

    # Créer la heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_delays, cmap='viridis', xticklabels=deadline_increments, yticklabels=False, cbar_kws={'label': 'Délai'})

    plt.xlabel('Augmentation de la deadline (delta)')
    plt.ylabel('Index des tâches')
    plt.title('Heatmap des délais pour chaque tâche en fonction de l\'augmentation de la deadline')
    plt.tight_layout()
    plt.show()


def main():
    # exps_time()
    exp_same_result()
    # exp_varying_deadlines()


if __name__ == "__main__":
    main()
