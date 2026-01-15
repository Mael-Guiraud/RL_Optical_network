import csv
import json
import multiprocessing
import os
from re import DEBUG
import time
# Importer le script ou appeler une fonction
from collections import Counter
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import lb_topo
import lb_traffic
from network import run_one_network
from p_dataset import get_instance_from_dataset
from Lien import *
from Paquet import *

import multiprocessing as mp
import sys
import time

_PROGRESS = None
_LOCK = None
_TOTAL = None
import os, sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            yield

def _init_progress(progress, lock, total):
    global _PROGRESS, _LOCK, _TOTAL
    _PROGRESS = progress
    _LOCK = lock
    _TOTAL = total

def _tick_progress():
    # appelé par les workers
    with _LOCK:
        _PROGRESS.value += 1

def run_all_algos(args):
    # ...
    for meta_algo in ["LRI_SEQUENTIEL","LRI"]:
        for submode in ["1", "2"]:
            for sort in ["sorted", "uniform"]:
                # ... ton code ...
                # à la fin de chaque combo:
                _tick_progress()

def generate_instance_and_traffic(nb_routes, nb_levels, period, c):
    instance = None
    while not instance:
        instance, nb_fails = lb_topo.generate_instance(nb_routes, nb_levels)
    if instance:
        # print(lb_topo.check_weakly_coherent(instance,nb_routes),lb_topo.check_strongly_coherent(instance,nb_routes))
        Fs, weight_matrix, _, _, _, B = lb_traffic.generate_random_traffic_no_save(instance, period, nb_routes, c)
    else:
        print("Error, no instance found")
        exit(1)
    return instance, Fs, weight_matrix, B



def find_action_index(paquet, delais_paquet):
    # Convertir les délais réels en indices par rapport à choix_possibles
    action_as_indices = [
        paquet.choix_possibles.tolist().index(delai) for delai in delais_paquet
    ]
    # Maintenant trouver l'index de cette action
    try:
        action_idx = paquet.actions.index(action_as_indices)
        return action_idx
    except ValueError:
        raise ValueError(f"Action {action_as_indices} non trouvée pour les délais {delais_paquet}")


def indicator_packet(packet: Paquet) -> int:
    """
    Returns 1 if ANY link used by `packet` is flagged either
        • collisions == True   OR
        • overflow   == True,
    otherwise 0.
    """
    for link in packet.liens_partages:
        if link.collisions or link.overflow:
            return 1
    return 0


def creer_paquets_depuis_instance(instance, Fs, weight_matrix, period, mct, B, delays=None, mode="LRI"):
    Paquets = {}
    nb_flows = len(instance[0])
    nb_paquets = [0 for _ in range(nb_flows)]

    for i in range(nb_flows):
        j = 0
        for o, k in enumerate(Fs[0][i][0][i]):
            if k != 0:
                depart_initial = (o + weight_matrix[0][i][0]) % period
                Paquets[(i, j)] = Paquet((i, j), len(instance) - 1, mct, period, depart_initial, initial_taumin=weight_matrix[0][i][0])
                j += 1
        nb_paquets[i] = j

    lien_partage = {}
    for i in range(nb_flows):
        for l in range(1, len(instance)):
            a = lb_traffic.find_id_router_and_cycle(instance, l, i)
            key = (l, a[0], a[1])
            if key not in lien_partage:
                lien_partage[key] = Lien(key, weight_matrix[l][a[0]][a[1]], B, period)
            lien = lien_partage[key]

            for j in range(nb_paquets[i]):
                lien.ajouter_paquet(Paquets[(i, j)])
                Paquets[(i, j)].liens_partages.append(lien)

    # Calculer taumin pour chaque paquet
    for p in Paquets.values():
        p.compute_taumin()

    # Initialiser les probas via delays si delays est fourni
    if delays is not None:
        for paquet in Paquets.values():
            delais_paquet = []
            route_id, _ = paquet.id_paquet
            t = paquet.depart_initial

            for lien in paquet.liens_partages:
                niveau, router, cycle = lien.id_lien
                delai = delays[niveau][router][cycle][route_id][int(t) % period]
                delais_paquet.append(int(delai))
                t = (t + delai + lien.poids) % period

            # Trouver l'action correspondant exactement aux délais
            try:
                action_idx = find_action_index(paquet, delais_paquet)
            except ValueError:
                # Si pas trouvé, on met une proba uniforme
                paquet.probas_actions = np.full(len(paquet.actions), 1 / len(paquet.actions))
                continue

            # Initialisation des probas : poids fort sur l'action trouvée
            if mode == "LRI" or mode == "LRI_SEQUENTIEL":
                paquet.probas_actions = np.full(len(paquet.actions), 0.01)
                paquet.probas_actions[action_idx] = 0.5
                paquet.probas_actions /= paquet.probas_actions.sum()
                paquet.delais = delais_paquet
            elif mode == "EXP3":
                paquet.init_exp3_weights(action_idx, preference_weight=10.0)
            else:
                print("Error 77: mode inconnu")
                exit(76)

    return Paquets, lien_partage


def test_coherence(paquets, liens, period):
    """
    Fait plusieurs tests de cohérence:
    1) Compare la présence dans presence_buffer (chaque lien) avec le calcul direct des horaires de paquets
    2) Vérifie la cohérence du calcul d'overflow
    3) Vérifie la cohérence du nombre de collisions
    Retourne True si tout semble correct, False sinon.
    """

    # Test 1 : Cohérence entre presence_buffer et horaires de paquets
    all_good = True

    for lien in liens.values():
        # Pour chaque slot, on va recalculer manuellement qui doit être dans le buffer
        recalcul = [set() for _ in range(period)]
        for paquet in lien.paquets:
            horaires_arr = paquet.horaires_arrivee()
            idx = lien.id_lien[0] - 1  # index du lien dans la route
            if idx < 0 or idx >= paquet.nb_liens:
                # Cas bizarre : on skip, ou on note un souci
                continue
            arrivee = horaires_arr[idx]
            delay = paquet.delais[idx]

            for t in range(arrivee, arrivee + delay):
                recalcul[t % period].add(paquet)

        # On compare recalcul[t] à lien.presence_buffer[t]
        for t_mod in range(period):
            # On construit un set() depuis le Counter
            slot_counter = lien.presence_buffer[t_mod]
            # On ne compare que les clés (présence / absence),
            # en ignorant le nombre exact d'occurrences.
            # Pour vérifier si le compte exact est bon, on peut comparer .items().
            presence_keys = {p for p, count in slot_counter.items() if count > 0}

            # si on veut comparer *strictement* le nombre d'occurrences,
            # on peut comparer slot_counter == some_other_counter

            if presence_keys != recalcul[t_mod]:
                print(f"[Incohérence] Lien {lien.id_lien}, slot={t_mod}")
                print(f"Présence selon presence_buffer: {presence_keys}")
                print(f"Présence recalculée         : {recalcul[t_mod]}")
                all_good = False

    # Test 2 : Vérifier la cohérence overflow
    # On recalcule l'occupation = somme des compteurs
    for lien in liens.values():
        for t_mod in range(period):
            effective_occupation = sum(lien.presence_buffer[t_mod].values())
            real_overflow = (effective_occupation > lien.buffer_max)
            method_overflow = lien.buffer_overflow(t_mod)
            if real_overflow != method_overflow:
                print(f"[Incohérence Overflow] Lien {lien.id_lien}, t={t_mod}")
                print(f"occupation={effective_occupation}, buffer_max={lien.buffer_max}")
                print(f"real_overflow={real_overflow}, method_overflow={method_overflow}")
                all_good = False

    # Test 3 : Vérifier collisions
    # On fait un double check sur un petit nombre de paquets (ou tous),
    # en recroisant horaires et en comparant avec .collisions()
    for p1 in paquets.values():
        collisions_calc = 0
        mes_horaires = p1.horaires_passage()
        for idx_lien, lien in enumerate(p1.liens_partages):
            # On check tous les voisins
            for p2 in lien.paquets:
                if p2 is not p1:
                    other_horaires = p2.horaires_passage()
                    # s'ils passent le même lien au même instant => collision
                    if other_horaires[idx_lien] == mes_horaires[idx_lien]:
                        collisions_calc += 1
        collisions_officiel = p1.collisions()
        if collisions_calc != collisions_officiel:
            print(f"[Incohérence Collisions] Paquet {p1.id_paquet}")
            print(f"collisions_calc={collisions_calc}, collisions_officiel={collisions_officiel}")
            all_good = False

    if all_good:
        print("\033[92m[OK] Tous les tests de cohérence semblent corrects.\033[0m")
    else:
        print("\033[91m[!] Des incohérences ont été détectées.\033[0m")

    return all_good


def total_overflow_count(paquet):
    """
    Calcule le nombre total d'instants où le paquet 'paquet'
    est en overflow sur l'un des liens qu'il parcourt.
    """
    count = 0
    for lien in paquet.liens_partages:
        count += lien.overflow_contribution(paquet)
    return count


def compute_mrs(paquet, collisions_j, overflow_j):
    """
    Si collisions_j == 0 et overflow_j == 0 :
       MRS = sum(paquet.delais) / paquet.taumin
    Sinon, MRS = paquet.compute_maxglob()
    """
    if paquet.taumin <= 0:
        print("Error 75 dans compute_mrs")
        exit(75)

    if collisions_j == 0 and overflow_j == 0:
        total_delay = sum(paquet.delais)
        return total_delay / paquet.taumin
    else:
        return paquet.compute_maxglob()


def compute_mdr(paquet):
    """
       Mdr = sum(paquet.delais) / paquet.taumin
    """
    if paquet.taumin <= 0:
        print("Error 75 dans compute_mdr")
        exit(75)
    total_delay = sum(paquet.delais)
    return total_delay / paquet.taumin

def compute_somme_delay(paquet):
    """
    Retourne la somme normalisée des délais pour le paquet :
        sum(paquet.delais) / paquet.taumin

    Cette valeur ne dépend pas des collisions/overflow et est utilisée
    notamment pour le mode *localinteraction*.
    """
    if paquet.taumin <= 0:
        print("Error 75 dans compute_somme_delay")
        exit(75)

    total_delay = sum(paquet.delais)
    return total_delay 


def compute_mrs_interaction(paquet, collisions_dict, overflow_dict):
    """
    Calcule le MRS du paquet comme étant le MRS maximal parmi tous les paquets
    partageant au moins un lien commun avec lui.

    Le MRS d'un paquet est :
    - MDR = sum(delais) / taumin si collisions = 0 et overflow = 0
    - Sinon, maxglob

    :param paquet: Le paquet considéré.
    :param collisions_dict: Dictionnaire {paquet: collisions}.
    :param overflow_dict: Dictionnaire {paquet: overflow}.
    """

    # Stocker le MRS maximal parmi les voisins de liens partagés
    max_mrs = 0.0

    # Inclure le paquet lui-même dans le calcul
    all_related_paquets = set()

    for lien in paquet.liens_partages:
        all_related_paquets.update(lien.get_paquets())

    for voisin in all_related_paquets:
        if voisin.taumin <= 0:
            print(f"Error: taumin non initialisé pour paquet {voisin.id_paquet}")
            exit(77)

        # Conditions pour calculer le MRS de chaque voisin
        if collisions_dict[voisin] == 0 and overflow_dict[voisin] == 0:
            mrs_voisin = sum(voisin.delais) / voisin.taumin
        else:
            mrs_voisin = voisin.compute_maxglob()

        # Stocker le max rencontré
        max_mrs = max(max_mrs, mrs_voisin)

    return max_mrs



def compute_cost_1(mdr_j, collisions_j, overflow_j, indicator_j,
                   max_mdr, max_coll, max_buff):
    """
    Cost with indicator I_j (0 or 1).

    ratio_delay = 1           si I_j = 1   (on pénalise au max)
                = mdr_j / max_mdr  sinon

    cost = 0.5 * ratio_delay
         + 0.25 * (collisions_j / max_coll)
         + 0.25 * (overflow_j   / max_buff)
    """
    # --- delay part --------------------------------------------------
    ratio_delay = indicator_j + (1 - indicator_j) * ((mdr_j ) / (max_mdr if max_mdr else 1))

    # --- safety divisors --------------------------------------------
    ratio_coll = collisions_j / max(1, max_coll)
    ratio_buff = overflow_j / max(1, max_buff)

    # --- final weighted cost ----------------------------------------
    return 0.5 * ratio_delay + 0.25 * ratio_coll + 0.25 * ratio_buff

def compute_cost_2(
    player,
    mdr_dict,
    collisions_j,
    overflow_j,
    max_coll,
    max_buff
):
    """
    Implémente :

    MDRj = 1 - (
        0.5 * ((mdr_j - MDR_min) / (MDR_max - MDR_min))
      + 0.25 * cols
      + 0.25 * buffer
    )
    """

    mdr_j = mdr_dict[player]
    MDR_min, MDR_max = get_interaction_mdr_bounds(player, mdr_dict)

    # Sécurité numérique
    if MDR_max == MDR_min:
        ratio_mdr = 0.0
    else:
        ratio_mdr = 1 - ((MDR_max - mdr_j) / (MDR_max - MDR_min))

    ratio_coll = collisions_j / max(1, max_coll)
    ratio_buff = overflow_j / max(1, max_buff)

    return 1.0 - (
        0.5 * ratio_mdr
        + 0.25 * ratio_coll
        + 0.25 * ratio_buff
    )


def cost_to_utility(cost_j, c_min, c_max):
    """
    utility(j) = (c_max - cost_j) / (c_max - c_min)
    """
    if c_max == c_min:
        return (c_max - cost_j)
    return (c_max - cost_j) / (c_max - c_min)


def create_probability_matrix(paquets):
    """
    Retourne une matrice 2D shape=(nb_joueurs, nb_actions).
    L'ordre des joueurs est l'ordre dans 'paquets.values()'.
    L'ordre des actions est l'indice standard 0..nb_actions-1.
    """
    # On trie éventuellement les paquets par leur id pour un affichage cohérent
    sorted_paquets = sorted(paquets.values(), key=lambda p: p.id_paquet)
    nb_joueurs = len(sorted_paquets)

    # Hypothèse: tous les paquets ont le même nombre d'actions
    nb_actions = len(sorted_paquets[0].actions)

    M = np.zeros((nb_joueurs, nb_actions))
    for i, pkt in enumerate(sorted_paquets):
        M[i, :] = pkt.probas_actions  # probas_actions shape=(nb_actions,)
    return M


def update_data(frame, paquets_history, im):
    """
    Fonction appelée à chaque frame de l'animation.
    - frame : indice de l'itération
    - paquets_history : liste de dictionnaires {id: paquet}, un par itération
    - im : l'objet imshow à mettre à jour
    """
    paquets_current = paquets_history[frame]
    M = create_probability_matrix(paquets_current)
    im.set_data(M)
    # Optionnel : actualiser l'échelle de couleurs si besoin
    # im.set_clim(0, 1)
    return [im]




def pick_next_player(
    remaining_players,
    state,
    selection_mode="sorted",  # "sorted" | "uniform"
):

    """
    Sélectionne le prochain joueur à jouer parmi remaining_players.

    selection_mode:
      - "sorted"  : invalides d'abord, MDR décroissant
      - "uniform" : tirage uniforme aléatoire
    """

    if selection_mode == "uniform":
        return random.choice(remaining_players)

    elif selection_mode == "sorted":
        valid = state["valid"]
        mdr = state["mdr"]

        return min(
            remaining_players,
            key=lambda p: (valid[p], -mdr[p])
        )

    else:
        raise ValueError(f"selection_mode inconnu: {selection_mode}")

def get_interaction_mdr_bounds(player, mdr_dict):
    """
    Retourne (MDR_min, MDR_max) parmi :
      - le joueur lui-même
      - tous les joueurs partageant au moins un lien avec lui
    """
    neighbours = set([player])
    for link in player.liens_partages:
        neighbours.update(link.paquets)

    values = [mdr_dict[p] for p in neighbours]

    return min(values), max(values)

def evaluate_state(paquets, liens, period, submode="selfish"):
    """
    Recalcule complètement l'état courant:
      - buffers
      - collisions / overflow
      - mdr_dict (MDR)
      - cost_dict (selon submode)
      - valid_dict (0/1)
    """
    # 1) reset + rebuild buffers
    for lien in liens.values():
        lien.reset_buffer()
    for lien in liens.values():
        lien.remplir_buffer()
    for link in liens.values():
        link.update_collisions()
        link.update_overflow()

    # 2) collisions/overflow/indicator
    collisions_dict = {}
    overflow_dict = {}
    indicator_dict = {}
    for p in paquets.values():
        collisions_dict[p] = p.collisions()
        overflow_dict[p] = total_overflow_count(p)
        indicator_dict[p] = indicator_packet(p)

    # 3) local maxima (pour localinteraction)
    local_max_coll = {}
    local_max_buff = {}
    for p in paquets.values():
        neighbours = set()
        for link in p.liens_partages:
            neighbours.update(link.paquets)
        local_max_coll[p] = max(collisions_dict[q] for q in neighbours) if neighbours else 0
        local_max_buff[p] = max(overflow_dict[q] for q in neighbours) if neighbours else 0

    max_coll = max(collisions_dict.values()) if collisions_dict else 0
    max_buff = max(overflow_dict.values()) if overflow_dict else 0

    # 4) MDR + (option interaction)
    mdr_dict = {}
    somme_delay_dict = {}

    for p in paquets.values():
        mdr_dict[p] = compute_mdr(p)
        somme_delay_dict[p] = compute_somme_delay(p)
    max_mdr = max(mdr_dict.values()) 


    # 5) costs
    cost_dict = {}
    for p in paquets.values():
        if submode == "1":
            c_j = compute_cost_1(mdr_dict[p], collisions_dict[p], overflow_dict[p], indicator_dict[p],
                                     max_mdr, local_max_coll[p], local_max_buff[p])
        elif submode == "2":
            c_j = compute_cost_2(player=p,mdr_dict=mdr_dict,collisions_j=collisions_dict[p],overflow_j=overflow_dict[p],max_coll=local_max_coll[p], max_buff=local_max_buff[p])
        else: #afficher message d'erreur et kill
            print("Error 78: submode inconnu dans evaluate_state:",submode)
            exit(78)
        cost_dict[p] = c_j

    # 6) validité
    valid_dict = {}
    for p in paquets.values():
        valid_dict[p] = 1 if (collisions_dict[p] == 0 and overflow_dict[p] == 0) else 0

    return {
        "collisions": collisions_dict,
        "overflow": overflow_dict,
        "indicator": indicator_dict,
        "mdr": mdr_dict,
        "cost": cost_dict,
        "valid": valid_dict,
        "max_coll": max_coll,
        "max_buff": max_buff,
    }
def pick_worst_player(remaining_players, valid_dict, mdr_dict):
    """
    remaining_players: iterable de Paquet
    Renvoie le paquet "pire" selon:
      1) invalides (valid=0) d'abord
      2) MDR décroissant
    """
    # valid=0 -> avant valid=1
    # MDR décroissant -> -mdr
    return min(
        remaining_players,
        key=lambda p: (valid_dict[p], -mdr_dict[p])
    )
def play_k_best_response(
    player,
    paquets,
    liens,
    period,
    K=10,
    submode="1",
):
    """
    Fait jouer `player` K fois, évalue après chaque tentative,
    et commit la meilleure action rencontrée.

    Retourne:
      best_action_idx, best_state, best_score
    """

    best_action_idx = None
    best_delays = None
    best_score = None  # score à minimiser

    for _ in range(K):
        action_idx = player.choisir_delais()

        # évaluation complète après cette action
        state = evaluate_state(paquets, liens, period, submode=submode)

        if submode == "1":
            score = state["cost"][player]
        elif submode == "2":
            score = 1-state["cost"][player]
        else:
            print("Error 78: submode inconnu dans evaluate_state:", submode)
            exit(78)

        # garder le meilleur (min score)
        if best_score is None or score < best_score:
            best_score = score
            best_action_idx = action_idx
            best_delays = player.delais.copy()

    # Commit: on remet les meilleurs délais trouvés
    player.delais = best_delays

    # Re-évaluer pour être cohérent avec l'état réellement commit
    # (optionnel mais recommandé, car on veut un state exact après commit)
    final_state = evaluate_state(paquets, liens, period, submode=submode)

    return best_action_idx, final_state

def run_learning_phases_sequential(
    paquets,
    liens,
    period,
    nb_phases=50,
    alpha=0.2,
    mode="LRI",
    submode="1",
    selection_mode="sorted"  # ou "uniform"
):
    """
    Version séquentielle par phases:
      - A chaque phase, tous les joueurs jouent une fois.
      - Ordre recalculé après chaque coup sur les joueurs restants:
          invalides d'abord (MDR décroissant), puis valides (MDR décroissant).
    Retour similaire:
      (best_value, avg_utility_valid_only, best_is_global_min_flag)
    """

  
    glob_s = []


    current_min = np.inf
    #calcul maxglob global (pour suivi trace)
    maxglob_max = 0
    for p in paquets.values():
        maxglob_max=max(maxglob_max, p.compute_maxglob())
    # Évaluation initiale (sert à initialiser le classement pour la phase 0)
    state = evaluate_state(paquets, liens, period, submode=submode)

    all_players = list(paquets.values())

    for phase in range(nb_phases):
        played = set()
        if __debug__:
            time.sleep(1)
            print("\n\n\n--- Nouvelle phase --- : ", phase)
        # Chaque phase: N coups
        while len(played) < len(all_players):
            remaining = [p for p in all_players if p not in played]
            if __debug__:
                print(f"Remaining players: {[p.id_paquet for p in remaining]}")
                print("State :",state["mdr"])
                print("\nValid :",state["valid"])
            # ordre basé sur l'état courant (recalculé après chaque coup)
            
            player = pick_next_player(
            remaining_players=remaining,
            state=state,
            selection_mode=selection_mode,  # paramètre
            )
            if __debug__:
                print(f"Player chosen to play: {player.id_paquet}")
            best_action_idx, state = play_k_best_response(
                player=player,
                paquets=paquets,
                liens=liens,
                period=period,
                K=10,
                submode=submode
            )
       

            # coût et utilité du joueur actif sur l'état commit
            c_j = state["cost"][player]
            player.update_costs_actions(best_action_idx, c_j)
            if submode == "1":
                u_j = cost_to_utility(
                    c_j,
                    player.min_previous_cost_action[best_action_idx],
                    player.max_previous_cost_action[best_action_idx],
                )
            else:
                u_j = c_j
            if __debug__:
                print(f"Player {player.id_paquet} played action {best_action_idx} with cost {c_j} and utility {u_j}")
            # update proba uniquement pour l'action retenue
            if mode == "LRI":
                player.update_probas_lri(best_action_idx, u_j, alpha)
            elif mode == "EXP3":
                player.update_probas_exp3(best_action_idx, u_j, gamma=alpha)
            else:
                print("Error: mode inconnu")
                exit(76)            
            played.add(player)

 
        
        if all(state["valid"].values()):
            phase_value = max(state["mdr"].values())
            glob_s.append(phase_value)
            if phase_value < current_min:
                current_min = phase_value
        else:
            glob_s.append(maxglob_max)
    

    return glob_s, current_min

def run_learning_iterations(paquets, liens, period, nb_iterations=10, alpha=0.2, mode="LRI"):
    glob_s = []
    max_utilities = []
    min_utilities = []

    valid_dict = {}
    
    #calcul maxglob global (pour suivi trace)
    maxglob_max = 0
    for p in paquets.values():
        maxglob_max=max(maxglob_max, p.compute_maxglob())
    current_min = np.inf
    average_utility = 0
    nb_utilities_values = 0

    for iteration in range(nb_iterations):
        for lien in liens.values():
            lien.reset_buffer()

        chosen_actions = {}
        for p in paquets.values():
            chosen_idx = p.choisir_delais()
            chosen_actions[p] = chosen_idx

        for lien in liens.values():
            lien.remplir_buffer()

        for link in liens.values():
            link.update_collisions()
            link.update_overflow()

        collisions_dict = {}
        overflow_dict = {}
        indicator_dict = {}
        mdr_dict = {}

        for p in paquets.values():
            collisions_dict[p] = p.collisions()
            overflow_dict[p] = total_overflow_count(p)
            indicator_dict[p] = indicator_packet(p)

        local_max_coll = {}
        local_max_buff = {}

        for p in paquets.values():
            neighbours = set()
            for link in p.liens_partages:
                neighbours.update(link.paquets)

            local_max_coll[p] = max(collisions_dict[q] for q in neighbours)
            local_max_buff[p] = max(overflow_dict[q] for q in neighbours)

        for p in paquets.values():
            mdr_dict[p] = compute_mdr(p)

        cost_dict = {}
        for p in paquets.values():
            c_j = compute_cost_1(mdr_dict[p], collisions_dict[p], overflow_dict[p], indicator_dict[p], max(mdr_dict.values()), local_max_coll[p], local_max_buff[p])
            cost_dict[p] = c_j
            p.update_costs_actions(chosen_actions[p], cost_dict[p])

        utility_dict = {}
        min_utility = 1
        max_utility = 0
        for p in paquets.values():
            utility_dict[p] = cost_to_utility(cost_dict[p], p.min_previous_cost_action[chosen_actions[p]], p.max_previous_cost_action[chosen_actions[p]])
            if collisions_dict[p] == 0 and overflow_dict[p] == 0:
                average_utility += utility_dict[p]
                nb_utilities_values += 1
            min_utility = min(min_utility, utility_dict[p])
            max_utility = max(max_utility, utility_dict[p])

        min_utilities.append(min_utility)
        max_utilities.append(max_utility)

        for p in paquets.values():
            valid_dict[p] = 1 if (collisions_dict[p] == 0 and overflow_dict[p] == 0) else 0
          
        if all(valid_dict):
            phase_value = max(mdr_dict.values())
            glob_s.append(phase_value)
            if phase_value < current_min:
                current_min = phase_value
        else:
            glob_s.append(maxglob_max)
    

      

        for p in paquets.values():
            chosen_idx = chosen_actions[p]
            if mode == "LRI":
                p.update_probas_lri(chosen_idx, utility_dict[p], alpha)
            elif mode == "EXP3":
                p.update_probas_exp3(chosen_idx, utility_dict[p], alpha)
            else:
                print("Error 76: mode inconnu")
                exit(76)

      
    return glob_s, current_min


def run_all_algos(args):
    for meta_algo in ["LRI_SEQUENTIEL","LRI"]:
        if meta_algo == "LRI":
            submode_list = ["1"]
            sort_list = ["sorted"]
        else:
            submode_list = ["1","2"]
            sort_list = ["sorted", "uniform"]
        for submode in submode_list:
            for sort in sort_list:

                i, instance, Fs, weight_matrix, nb_routes, period, mct, B, timestamp, nb_levels = args

                start = time.time()
                with suppress_output():
                    poly_value, delays, _ = run_one_network(instance, Fs, weight_matrix, nb_routes, period, mct, B, mode="POLYOPTI")
                poly_computation_time = time.time() - start
                if poly_value == -1:
                    delays = None
                algo = "LRI"
                


                # Créer une nouvelle instance de paquets et de liens pour chaque exécution
                paquets, liens = creer_paquets_depuis_instance(instance, Fs, weight_matrix, period, mct, B, delays, mode=algo)
                start = time.time()
                if meta_algo == "LRI":
                    glob_s,glob_s_min = run_learning_iterations(paquets, liens, period, nb_iterations=50000, alpha=0.001, mode=algo)
                else:
                    glob_s,glob_s_min = run_learning_phases_sequential(
                        paquets, liens, period,
                        nb_phases=50000//len(paquets),  # 50000/nombre de joueurs
                        alpha=0.001,
                        mode=algo,
                        submode=submode,
                        selection_mode=sort  # "sorted" ou "random"
                    )
                RL_computation_time = time.time() - start
            

                # Écrire les résultats dans un fichier CSV spécifique au thread
                filename = f"results_r{nb_routes}_l{nb_levels}_p{period}_c{mct}_i{i}.csv"
                result = {
                    "algo": meta_algo,
                    "submode": submode,
                    "sort": sort,
                    "run": i,
                    "glob_s_min": glob_s_min,
                    "poly_value": poly_value,
                    "poly_computation_time": poly_computation_time,
                    "RL_computation_time": RL_computation_time,
                }

                # Créer les répertoires si nécessaire
                if not os.path.exists("./results"):
                    os.makedirs("./results")
                if not os.path.exists("./results/RL"):
                    os.makedirs("./results/RL")
                folder = os.path.join("./results/RL", timestamp)
                if not os.path.exists(folder):
                    os.makedirs(folder)

                file_path = os.path.join(folder, filename)

                # Déterminer si le fichier existe déjà pour écrire les en-têtes
                file_exists = os.path.isfile(file_path)

                with open(file_path, "a", newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=result.keys())
                    if not file_exists:
                        writer.writeheader()  # Écrire l'en-tête seulement si le fichier n'existe pas
                    writer.writerow(result)
                _tick_progress()



if __name__ == "__main__":
    nb_tests = 100
    total = nb_tests * 5
    nb_routes = 5
    nb_levels = 2
    period = 10
    mct = 2
    B = 3

    
    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S")
   
    args_list = []
    for i in range(nb_tests):

        data = get_instance_from_dataset(nb_routes, nb_levels, period, mct, i)
        instance, S, Fs, weight_matrix = data[4], data[5], data[6], data[7]
        args_list.append((i, instance, Fs, weight_matrix, nb_routes, period, mct, B, timestamp, nb_levels))

    # Utiliser Pool pour paralléliser les tâches et limiter le nombre de cœurs
    num_cores = max(1, multiprocessing.cpu_count())  # Limitez le nombre de cœurs à utiliser
    num_cores = min(num_cores,nb_tests)
    with mp.Manager() as mgr:
        progress = mgr.Value('i', 0)
        lock = mgr.Lock()

        with Pool(
            processes=num_cores,
            initializer=_init_progress,
            initargs=(progress, lock, total),
        ) as pool:

            # imap_unordered permet au parent de reprendre la main et d'afficher
            it = pool.imap_unordered(run_all_algos, args_list)

            start = time.time()
            completed_tests = 0

            for _ in it:
                completed_tests += 1

                done = progress.value
                elapsed = max(1e-9, time.time() - start)
                rate = done / elapsed
                remaining = total - done
                eta = remaining / rate if rate > 0 else float("inf")

                sys.stdout.write(
                    f"\rProgress: {done}/{total} combos | "
                    f"{completed_tests}/{nb_tests} tests | "
                    f"{rate:.2f} combos/s | ETA ~ {eta:.0f}s"
                )
                sys.stdout.flush()

        print()  # retour à la ligne à la fin