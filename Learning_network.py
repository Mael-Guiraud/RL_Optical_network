import csv
import json
import multiprocessing
import os
import time
# Importer le script ou appeler une fonction
from collections import Counter
from functools import lru_cache
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

import lb_topo
import lb_traffic
from network import run_one_network
from p_dataset import get_instance_from_dataset


class Lien:
    def __init__(self, id_lien, poids, buffer_max, period):
        self.id_lien = id_lien
        self.poids = poids
        self.buffer_max = buffer_max
        self.period = period
        self.collisions = False
        self.overflow = False
        # Liste de taille period, chaque case étant un Counter
        # Key = paquet, Value = nombre d'occurrences
        self.presence_buffer = [Counter() for _ in range(self.period)]
        self.paquets = set()

    def reset_buffer(self):
        for slot in range(self.period):
            self.presence_buffer[slot].clear()

    def remplir_buffer_paquet(self, paquet):
        """
        Si un paquet dépasse de plus d'une période (delay > P),
        on comptera autant d'occurrences de ce paquet dans le même slot.
        """
        horaires = paquet.horaires_arrivee()
        idx = self.id_lien[0] - 1  # exemple d'indexation : à adapter à tes besoins
        arrivee = horaires[idx]
        delay = paquet.delais[idx]

        for t in range(arrivee, arrivee + delay):
            self.presence_buffer[t % self.period][paquet] += 1

    def remplir_buffer(self):
        for paquet in self.paquets:
            self.remplir_buffer_paquet(paquet)

    def occupation_buffer(self, t):
        """
        Retourne le nombre total d'occurrences (somme des compteurs)
        pour tous les paquets présents dans le buffer à l’instant t.
        """
        return sum(self.presence_buffer[t % self.period].values())

    def buffer_overflow(self, t):
        return self.occupation_buffer(t) > self.buffer_max

    def update_collisions(self) -> None:
        """
        Sets self.collisions = True if at least two packets
        use the same time-slot on this link; otherwise False.
        """
        self.collisions = False  # reset
        slot_seen = {}  # slot -> first packet

        for pkt in self.paquets:
            # index of this link in packet's route
            try:
                idx = pkt.liens_partages.index(self)
            except ValueError:
                continue  # should not happen

            slot = pkt.horaires_passage()[idx]  # slot when pkt uses link

            if slot in slot_seen:  # second packet on same slot
                self.collisions = True
                return  # early exit
            slot_seen[slot] = pkt

    def update_overflow(self):
        """
        Sets self.overflow = True if at least one time-slot
        has more packets than buffer_max; otherwise False.
        """
        self.overflow = False  # reset

        for t_mod in range(self.period):
            if self.occupation_buffer(t_mod) > self.buffer_max:
                self.overflow = True
                return  # early exit

    def get_collisions(self):
        return self.collisions

    def get_overflow(self):
        return self.overflow

    def overflow_contribution(self, paquet):
        """
        Renvoie le nombre de tics (0..period-1) sur lesquels le paquet 'paquet'
        a au moins 1 occurrence ET où le buffer dépasse buffer_max.
        """
        count = 0
        for t_mod in range(self.period):
            if self.presence_buffer[t_mod][paquet] > 0:  # le paquet est présent
                occ = self.occupation_buffer(t_mod)
                if occ > self.buffer_max:
                    count += self.presence_buffer[t_mod][paquet] * max(occ - self.buffer_max, 0)
        return count

    def print_buffer(self):
        for slot in range(self.period):
            print(f"Slot {slot} : {self.presence_buffer[slot]}")

    def ajouter_paquet(self, paquet):
        self.paquets.add(paquet)

    def retirer_paquet(self, paquet):
        self.paquets.discard(paquet)

    def __eq__(self, other):
        return (self.id_lien[0], self.id_lien[1], self.id_lien[2]) == (other.id_lien[0], other.id_lien[1], other.id_lien[2])

    def __hash__(self):
        return self.id_lien[0] * 100 + self.id_lien[1] * 10 + self.id_lien[2]

    def get_paquets(self):
        return self.paquets

    def __repr__(self):
        return f"Lien({self.id_lien}, poids={self.poids}, buffer_max={self.buffer_max})"


def reset_schedule(f):
    for i, level in enumerate(f):
        if i > 0:
            for j, router in enumerate(level):
                for k, cycle in enumerate(router):
                    for l, route in enumerate(cycle):
                        for m, tic in enumerate(route):
                            f[i][j][k][l][m] = 0


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


@lru_cache(maxsize=None)
def generate_arrangement_cached(N, P):
    if N == 0:
        return [()]
    else:
        result = []
        sub_words = generate_arrangement_cached(N - 1, P)
        for i in range(P):
            for sub in sub_words:
                result.append((i,) + sub)
        return tuple(result)


def generate_arrangement(N, P):
    cached_result = generate_arrangement_cached(N, P)
    return [list(x) for x in cached_result]


class Paquet:
    def __init__(self, id_paquet, nb_liens, mct, P, depart_initial, delais_initiaux=None, initial_taumin=0):
        """
        Initialise un objet Paquet.

        :param id_paquet: Identifiant unique du paquet.
        :param nb_liens: Nombre de liens traversés par le paquet.
        :param liens_partages: Liste de sets des autres paquets partageant chaque lien.
        :param mct: Minimal Conversion Time.
        :param P: Période.
        :param delais_initiaux: Délais initiaux choisis (par défaut tous 0).
        """
        self.id_paquet = id_paquet
        self.nb_liens = nb_liens
        self.liens_partages = []  # List[Set[Paquet]]
        self.P = P
        self.mct = mct
        self.depart_initial = depart_initial
        self.taumin = initial_taumin
        # Ensemble des choix possibles pour chaque lien
        self.choix_possibles = np.concatenate(([0], np.arange(mct, P)))
        # self.choix_possibles = np.concatenate(([0], np.arange(mct, P), np.arange(P + 1, P + mct)))
        self.nb_choix = len(self.choix_possibles)  # Exactement P

        self.actions = generate_arrangement(nb_liens, self.nb_choix)
        # Matrice de probabilités initialement uniforme (nb_liens x nb_choix)
        self.probas_actions = np.full(len(self.actions), 1 / len(self.actions))
        self.max_previous_cost_action = [0.4 for _ in range(len(self.actions))]
        self.min_previous_cost_action = [0.4 for _ in range(len(self.actions))]
        # Délais initiaux
        if delais_initiaux:
            self.delais = delais_initiaux
        else:
            self.delais = [0 for _ in range(nb_liens)]

    def update_costs_actions(self, action_idx, cost):
        """
        Met à jour le coût de l'action action_idx.
        """
        self.max_previous_cost_action[action_idx] = max(cost, self.max_previous_cost_action[action_idx])
        self.min_previous_cost_action[action_idx] = min(cost, self.min_previous_cost_action[action_idx])

    def compute_taumin(self):
        self.taumin += sum(lien.poids for lien in self.liens_partages)

    def choisir_delais(self):
        """
        Choisit un délai pour chaque lien selon les probabilités actuelles.
        """

        choix = np.random.choice(len(self.actions), p=self.probas_actions)
        for lien in range(self.nb_liens):
            self.delais[lien] = self.choix_possibles[self.actions[choix][lien]]
        return choix

    def update_probas_lri(self, chosen_action_idx, utility, alpha=0.1):
        """
        Mise à jour LRI :
        - chosen_action_idx : index de l'action jouée (i.e. l'index dans self.actions).
        - utility : un scalaire dans [0..1].
        - alpha : taux d'apprentissage.
        """
        p_chosen = self.probas_actions[chosen_action_idx]

        # On augmente la proba de l'action choisie proportionnellement à utility
        self.probas_actions[chosen_action_idx] = p_chosen + alpha * utility * (1 - p_chosen)

        # On réduit les autres
        for i in range(len(self.probas_actions)):
            if i != chosen_action_idx:
                self.probas_actions[i] *= (1 - alpha * utility)

        # Normalisation éventuelle
        s = sum(self.probas_actions)
        if s > 0:
            self.probas_actions = self.probas_actions / s

    def horaires_arrivee(self):
        horaires = []
        t = self.depart_initial
        for idx, lien in enumerate(self.liens_partages):
            horaires.append(t)
            t += self.delais[idx]
            t += lien.poids
            t = t % self.P
        return horaires

    def horaires_passage(self):
        horaires = []
        t = self.depart_initial
        for idx, lien in enumerate(self.liens_partages):
            t += self.delais[idx]
            t = t % self.P
            horaires.append(t)
            t += lien.poids
            t = t % self.P
        return horaires

    def compute_maxglob(self):
        """
        Calcule la borne supérieure de MRS pour CE paquet.
        MRS max = (nb_liens * (P + mct - 1)) / taumin
        """
        if self.taumin <= 0:
            print("Error 74 dans compute_maxglob: taumin pas initialisé ? Valeur de taumin :", self.taumin)
            exit(74)
        return (self.nb_liens * (self.P + self.mct - 1)) / self.taumin

    def collisions(self):
        mes_horaires = [t % self.P for t in self.horaires_passage()]
        nb_collisions = 0
        for idx_lien, lien in enumerate(self.liens_partages):
            voisins = lien.get_paquets()
            for voisin in voisins:
                if voisin.id_paquet != self.id_paquet:  # ne pas compter soi-même
                    horaires_voisin = [t % self.P for t in voisin.horaires_passage()]
                    if horaires_voisin[idx_lien] == mes_horaires[idx_lien]:
                        nb_collisions += 1
        return nb_collisions

    def temps_total(self):
        return sum(self.temps_liens) + sum(self.delais)

    def delai_max(self):
        return sum(self.delais)

    def __repr__(self):
        return f"[Paquet({self.id_paquet})]"

    def update_probas_exp3(self, chosen_action_idx, reward, gamma=0.07):
        """
        Mise à jour des probabilités via l'algorithme Exp3.
        - chosen_action_idx : index de l'action jouée
        - reward : retour (scalaire ∈ [0,1])
        - gamma : paramètre d'exploration
        """
        K = len(self.probas_actions)

        # Initialisation des poids si nécessaire
        if not hasattr(self, 'weights') or len(self.weights) != K:
            self.weights = np.ones(K)

        # Probabilité de l'action jouée (évite division par 0)
        p_chosen = max(self.probas_actions[chosen_action_idx], 1e-12)

        # Gain estimé
        G_hat = reward / p_chosen

        # Mise à jour du poids de l'action choisie (stabilisation du produit exp)
        log_update = gamma * G_hat / K
        log_update = np.clip(log_update, -100, 100)  # évite overflow
        self.weights[chosen_action_idx] *= np.exp(log_update)

        # Renormalisation stable
        total_weight = np.sum(self.weights)
        if total_weight <= 0 or np.isnan(total_weight) or np.isinf(total_weight):
            self.weights = np.ones(K)
            total_weight = K

        # Mise à jour des probabilités
        self.probas_actions = (1 - gamma) * (self.weights / total_weight) + (gamma / K)

        # Clip final + normalisation défensive
        self.probas_actions = np.clip(self.probas_actions, 1e-12, 1)
        self.probas_actions /= np.sum(self.probas_actions)

    def init_exp3_weights(self, preferred_action_idx, preference_weight=10.0):
        K = len(self.actions)
        # Initialiser tous les poids à 1 (ou à une valeur faible/uniforme)
        self.weights = np.ones(K)

        # Augmenter le poids de l'action préférée
        self.weights[preferred_action_idx] = preference_weight

        # Mettre à jour les probabilités en fonction des nouveaux poids
        total_weight = np.sum(self.weights)
        gamma = 0.1  # le gamma que tu utiliseras pour exp3
        self.probas_actions = (1 - gamma) * (self.weights / total_weight) + (gamma / K)


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
            if mode == "LRI":
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


def compute_somme_delay(paquet):
    """
    Si collisions_j == 0 et overflow_j == 0 :
       MRS = sum(paquet.delais) / paquet.taumin
    Sinon, MRS = paquet.compute_maxglob()
    """
    if paquet.taumin <= 0:
        print("Error 75 dans compute_mrs")
        exit(75)
    return paquet.compute_maxglob()


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


def compute_cost(mrs_j, collisions_j, overflow_j, max_glob_local, max_coll, max_buff):
    """
    cost_j = 0.5 * (mrs_j / max_glob_local)
             + 0.25 * (collisions_j / max_coll)
             + 0.25 * (overflow_j / max_buff)
    """
    ratio_mrs = (mrs_j / max_glob_local)
    ratio_coll = (collisions_j / max(1, max_coll))
    ratio_buff = (overflow_j / max(1, max_buff))

    return 0.2 * ratio_mrs + 0.4 * ratio_coll + 0.4 * ratio_buff


def compute_cost_2(mrs_j, collisions_j, overflow_j, indicator_j,
                   max_glob_local, max_coll, max_buff):
    """
    Cost with indicator I_j (0 or 1).

    ratio_delay = 1           si I_j = 1   (on pénalise au max)
                = mrs_j / max_glob_local  sinon

    cost = 0.5 * ratio_delay
         + 0.25 * (collisions_j / max_coll)
         + 0.25 * (overflow_j   / max_buff)
    """
    # --- delay part --------------------------------------------------
    ratio_delay = indicator_j + (1 - indicator_j) * ((mrs_j ) / max_glob_local)

    # --- safety divisors --------------------------------------------
    ratio_coll = collisions_j / max(1, max_coll)
    ratio_buff = overflow_j / max(1, max_buff)

    # --- final weighted cost ----------------------------------------
    return 0.5 * ratio_delay + 0.25 * ratio_coll + 0.25 * ratio_buff


def compute_cost_interaction(mrs_j, collisions_j, overflow_j, max_glob_local, max_coll, max_buff):
    """
    cost_j = 0.5 * (mrs_j / max_glob_local)
             + 0.25 * (collisions_j / max_coll)
             + 0.25 * (overflow_j / max_buff)
    """
    ratio_mrs = (mrs_j / 2 * max_glob_local)
    ratio_coll = (collisions_j / max(1, max_coll))
    ratio_buff = (overflow_j / max(1, max_buff))

    return 0.5 * ratio_mrs + 0.25 * ratio_coll + 0.25 * ratio_buff

def compute_cost_mrs_interaction(
    player,
    mrs_dict,
    collisions_dict,
    overflow_dict,
):
    """
    Implémente :

    MRSj = 1 - (
        0.5 * ((mdr_j - MDR_min) / (MDR_max - MDR_min))
      + 0.25 * cols
      + 0.25 * buffer
    )
    """

    mdr_j = mrs_dict[player]
    MDR_min, MDR_max = get_interaction_mdr_bounds(player, mrs_dict)

    # Sécurité numérique
    if MDR_max == MDR_min:
        ratio_mdr = 0.0
    else:
        ratio_mdr = (mdr_j - MDR_min) / (MDR_max - MDR_min)

    cols = collisions_dict[player]
    buffer = overflow_dict[player]

    return 1.0 - (
        0.5 * ratio_mdr
        + 0.25 * cols
        + 0.25 * buffer
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


def animate_probabilities(probas_history):
    """
    probas_history est une liste de matrices (np.array)
    shape=(nb_joueurs, nb_actions) pour chaque itération
    """
    fig, ax = plt.subplots()

    im = ax.imshow(probas_history[0], aspect='auto', origin='upper',
                   vmin=0, vmax=1, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Actions")
    ax.set_ylabel("Joueurs")
    ax.set_title("Iteration 0")

    def update(frame):
        im.set_data(probas_history[frame])
        ax.set_title(f"Iteration {frame * 100}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(probas_history),
                        interval=500, blit=True)

    return ani


def save_learning_results(it_number, algo_name, period, glob_s, glob_s_min, valid, max_utilities, min_utilities, average_utility_list, poly_value, probas_history, ani, timestamp, base_dir, ax):
    # Sauvegarde courbe MRS et utilités sur les axes fournis
    ax[0].plot(glob_s, label="Glob S")
    ax[0].plot(glob_s_min, label="Min MRS")
    if poly_value != -1:
        ax[0].axhline(y=poly_value, color='red', linestyle='--', label='LP couche')
    else:
        print(f"Aucune valeur poly_value fournie, pas de ligne rouge pour itération {it_number}")
    ax[0].set_title("Évolution du max(MRS) au fil des itérations")
    ax[0].set_xlabel("Itération")
    ax[0].set_ylabel("Valeur MRS")
    ax[0].legend(loc="upper right")

    ax[1].plot(max_utilities, label="Utilité max")
    ax[1].plot(min_utilities, label="Utilité min")
    ax[1].plot(average_utility_list, label="Utilité moyenne")
    ax[1].set_title("Évolution des utilités")
    ax[1].set_xlabel("Itération")
    ax[1].set_ylabel("Utilité")
    ax[1].legend(loc="upper right")
import random

def pick_next_player(
    remaining_players,
    state,
    selection_mode="worst",  # "worst" | "random"
):

    """
    Sélectionne le prochain joueur à jouer parmi remaining_players.

    selection_mode:
      - "worst"  : invalides d'abord, MDR décroissant
      - "random" : tirage uniforme aléatoire
    """

    if selection_mode == "uniform":
        return random.choice(remaining_players)

    elif selection_mode == "sorted":
        valid = state["valid"]
        mrs = state["mrs"]

        return min(
            remaining_players,
            key=lambda p: (valid[p], -mrs[p])
        )

    else:
        raise ValueError(f"selection_mode inconnu: {selection_mode}")

def get_interaction_mdr_bounds(player, mrs_dict):
    """
    Retourne (MDR_min, MDR_max) parmi :
      - le joueur lui-même
      - tous les joueurs partageant au moins un lien avec lui
    """
    neighbours = set([player])
    for link in player.liens_partages:
        neighbours.update(link.paquets)

    values = [mrs_dict[p] for p in neighbours]

    return min(values), max(values)

def evaluate_state(paquets, liens, period, submode="selfish"):
    """
    Recalcule complètement l'état courant:
      - buffers
      - collisions / overflow
      - mrs_dict (MDR)
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
    mrs_dict = {}
    mrs_interaction_dict = {}
    somme_delay_dict = {}

    for p in paquets.values():
        mrs_dict[p] = compute_mrs(p, collisions_dict[p], overflow_dict[p])
        somme_delay_dict[p] = compute_somme_delay(p)
        if submode != "selfish":
            mrs_interaction_dict[p] = 2 * compute_mrs_interaction(p, collisions_dict, overflow_dict) - mrs_dict[p]

    # 5) costs
    cost_dict = {}
    for p in paquets.values():
        if submode == "1":
            c_j = compute_cost_2(somme_delay_dict[p], collisions_dict[p], overflow_dict[p], indicator_dict[p],
                                     p.compute_maxglob(), local_max_coll[p], local_max_buff[p])
        elif submode == "2":
            c_j = compute_cost_mrs_interaction(
                player=p,
                mrs_dict=mrs_dict,
                collisions_dict=collisions_dict,
                overflow_dict=overflow_dict,
            )
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
        "mrs": mrs_dict,
        "cost": cost_dict,
        "valid": valid_dict,
        "max_coll": max_coll,
        "max_buff": max_buff,
    }
def pick_worst_player(remaining_players, valid_dict, mrs_dict):
    """
    remaining_players: iterable de Paquet
    Renvoie le paquet "pire" selon:
      1) invalides (valid=0) d'abord
      2) MDR décroissant
    """
    # valid=0 -> avant valid=1
    # MDR décroissant -> -mrs
    return min(
        remaining_players,
        key=lambda p: (valid_dict[p], -mrs_dict[p])
    )
def play_k_best_response(
    player,
    paquets,
    liens,
    period,
    K=10,
    submode="selfish",
    selection_metric="player_cost",
):
    """
    Fait jouer `player` K fois, évalue après chaque tentative,
    et commit la meilleure action rencontrée.

    Retourne:
      best_action_idx, best_state, best_score
    """

    # Sauvegarde des délais initiaux pour pouvoir tester sans perdre l'état
    original_delays = player.delais.copy()

    best_action_idx = None
    best_delays = None
    best_state = None
    best_score = None  # score à minimiser

    for _ in range(K):
        action_idx = player.choisir_delais()

        # évaluation complète après cette action
        state = evaluate_state(paquets, liens, period, submode=submode)

        # score selon le critère choisi
        if selection_metric == "player_cost":
            score = state["cost"][player]
        elif selection_metric == "global_mdr":
            score = max(state["mrs"].values()) if state["mrs"] else 0.0
        else:
            raise ValueError("selection_metric inconnu")

        # garder le meilleur (min score)
        if best_score is None or score < best_score:
            best_score = score
            best_action_idx = action_idx
            best_delays = player.delais.copy()
            best_state = state

    # Commit: on remet les meilleurs délais trouvés
    player.delais = best_delays

    # Re-évaluer pour être cohérent avec l'état réellement commit
    # (optionnel mais recommandé, car on veut un state exact après commit)
    final_state = evaluate_state(paquets, liens, period, submode=submode)

    return best_action_idx, final_state, best_score

def run_learning_phases_sequential(
    paquets,
    liens,
    period,
    nb_phases=50,
    alpha=0.2,
    poly_value=0,
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

    glob_s_min = []
    glob_s = []
    valid_history = []

    best_delay = None
    current_min = np.inf

    average_utility = 0.0
    nb_utilities_values = 0

    # Évaluation initiale (sert à initialiser le classement pour la phase 0)
    state = evaluate_state(paquets, liens, period, submode=submode)

    all_players = list(paquets.values())

    for phase in range(nb_phases):
        played = set()

        # Chaque phase: N coups
        while len(played) < len(all_players):
            remaining = [p for p in all_players if p not in played]

            # ordre basé sur l'état courant (recalculé après chaque coup)
         
            player = pick_next_player(
            remaining_players=remaining,
            state=state,
            selection_mode=selection_mode,  # paramètre
            )
            best_action_idx, state, best_score = play_k_best_response(
                player=player,
                paquets=paquets,
                liens=liens,
                period=period,
                K=10,
                submode=submode,
                selection_metric="player_cost",  # ou "global_mdr"
            )

            # coût et utilité du joueur actif sur l'état commit
            c_j = state["cost"][player]
            player.update_costs_actions(best_action_idx, c_j)

            u_j = cost_to_utility(
                c_j,
                player.min_previous_cost_action[best_action_idx],
                player.max_previous_cost_action[best_action_idx],
            )

            # update proba uniquement pour l'action retenue
            if mode == "LRI":
                player.update_probas_lri(best_action_idx, u_j, alpha)
            elif mode == "EXP3":
                player.update_probas_exp3(best_action_idx, u_j, gamma=alpha)
            else:
                print("Error: mode inconnu")
                exit(76)

            played.add(player)

            
            played.add(player)

        # Fin de phase: métriques globales
        # (tu peux prendre max MDR, comme dans ton code)
        phase_val = max(state["mrs"].values()) if state["mrs"] else 0.0
        glob_s.append(phase_val)

        if phase_val < current_min:
            current_min = phase_val
            best_delay = {pid: p.delais.copy() for pid, p in paquets.items()}

        glob_s_min.append(current_min)

        # Validité globale (aucune collision/overflow sur tous)
        # Ici on suit ta logique: valid=1 si TOUT le monde est valide
        valid_history.append(1 if all(state["valid"].values()) else 0)

    if nb_utilities_values > 0:
        avg_u = average_utility / nb_utilities_values
    else:
        avg_u = 0.0

    # Même convention que toi
    best_is_global_min_flag = (glob_s_min[-1] == max(glob_s_min)) if glob_s_min else False

    # Optionnel: logs comparatif LP
    if glob_s_min and poly_value != -1 and glob_s_min[-1] < poly_value:
        print("Séquentiel meilleur que LP", submode, ": RL", glob_s_min[-1], "LP:", poly_value)

    return glob_s_min[-1], avg_u, best_is_global_min_flag

def run_learning_iterations(paquets, liens, period, nb_iterations=10, alpha=0.2, poly_value=0, mode="LRI", submode="selfish", it_number=0, Save=1, timestamp=None, base_dir=None, ax=None):
    glob_s = []
    max_utilities = []
    min_utilities = []
    glob_s_min = []
    probas_history = []
    valid = []
    best_delay = None
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
        mrs_dict = {}
        mrs_interaction_dict = {}
        indicator_dict = {}
        somme_delay_dict = {}

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

        max_coll = max(collisions_dict.values()) if collisions_dict else 0
        max_buff = max(overflow_dict.values()) if overflow_dict else 0

        for p in paquets.values():
            mrs_dict[p] = compute_mrs(p, collisions_dict[p], overflow_dict[p])
            somme_delay_dict[p] = compute_somme_delay(p)
            if submode != "selfish":
                mrs_interaction_dict[p] = 2 * compute_mrs_interaction(p, collisions_dict, overflow_dict) - mrs_dict[p]

        cost_dict = {}
        for p in paquets.values():
            if submode == "selfish":
                c_j = compute_cost(mrs_dict[p], collisions_dict[p], overflow_dict[p], p.compute_maxglob(), max_coll, max_buff)
            else:
                if submode == "localinteraction":
                    #c_j = compute_cost_2(mrs_dict[p], collisions_dict[p], overflow_dict[p], indicator_dict[p], p.compute_maxglob(), local_max_coll[p], local_max_buff[p])
                    c_j = compute_cost_2(somme_delay_dict[p], collisions_dict[p], overflow_dict[p], indicator_dict[p], p.compute_maxglob(), local_max_coll[p], local_max_buff[p])
                else:
                    c_j = compute_cost_interaction(mrs_interaction_dict[p], collisions_dict[p], overflow_dict[p], p.compute_maxglob(), local_max_coll[p], local_max_buff[p])
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
        val = max(mrs_dict.values())
        glob_s.append(val)

        if val < current_min:
            current_min = val
            best_delay = {pid: p.delais.copy() for pid, p in paquets.items()}

        glob_s_min.append(current_min)

        for p in paquets.values():
            chosen_idx = chosen_actions[p]
            if mode == "LRI":
                p.update_probas_lri(chosen_idx, utility_dict[p], alpha)
            elif mode == "EXP3":
                p.update_probas_exp3(chosen_idx, utility_dict[p], alpha)
            else:
                print("Error 76: mode inconnu")
                exit(76)

        if sum(collisions_dict.values()) == 0 and sum(overflow_dict.values()) == 0:
            valid.append(1)
        else:
            valid.append(0)

        if iteration % 1000 == 0:
            M = create_probability_matrix(paquets)
            probas_history.append(M.copy())

    if glob_s_min[-1] < poly_value:
        print("LRI meilleur que LP", submode, ": LRI ", glob_s_min[-1], "LP :", poly_value)
    if poly_value == -1 and glob_s_min[-1] < max(glob_s):
        print("LRI meilleur que LP", submode, ": LRI ", glob_s_min[-1], "LP : non calculé")

    # if Save:
    #    average_utility_list = np.cumsum([u for u in max_utilities]) / (np.arange(len(max_utilities)) + 1)
    #    algo_name = mode + ("_" + submode if mode == "LRI" else "")
    #    save_learning_results(it_number, algo_name, period, glob_s, glob_s_min, valid, max_utilities, min_utilities, average_utility_list, poly_value, probas_history, 0, timestamp, base_dir, ax)

    return glob_s_min[-1], average_utility / nb_utilities_values, glob_s_min[-1] == max(glob_s_min)


def run_all_algos(args):
    i, instance, Fs, weight_matrix, nb_routes, period, mct, B, timestamp, base_dir = args
    print("Running all algorithms for test", i)

    poly_value, delays, _ = run_one_network(instance, Fs, weight_matrix, nb_routes, period, mct, B, mode="POLYOPTI")
    if poly_value == -1:
        delays = None

    algo = "LRI"
    # Créer une figure avec trois sous-graphiques
    fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    min_glob_s_min = np.inf
    for run in range(3):
        # Créer une nouvelle instance de paquets et de liens pour chaque exécution
        paquets, liens = creer_paquets_depuis_instance(instance, Fs, weight_matrix, period, mct, B, delays, mode=algo)

        glob_s_min, average_utility, fail = run_learning_iterations(paquets, liens, period, nb_iterations=80000, alpha=0.001, poly_value=poly_value, mode=algo, submode="localinteraction", it_number=i, timestamp=timestamp, base_dir=base_dir, ax=axs[run])
        if glob_s_min < min_glob_s_min:
            min_glob_s_min = glob_s_min

    # Sauvegarder la figure avec les trois sous-graphiques
    output_dir = os.path.join(base_dir, str(algo))
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"metrics_run_{i}.png"))
    plt.close()

    # Écrire les résultats dans un fichier temporaire spécifique au thread
    temp_file_path = f"temp_results_{i}.json"
    result = {
        "algo": algo,
        "run": i,
        "glob_s_min": min_glob_s_min,
        "poly_value": poly_value,
        "average_utility": average_utility
    }

    with open(temp_file_path, "a") as f:
        json.dump(result, f)
        f.write("\n")



def run_all_algos_no_fig(args):
    for submode in ["1", "2"]:
        for sort in ["sorted", "uniform"]:
            
            i, instance, Fs, weight_matrix, nb_routes, period, mct, B, timestamp, nb_levels = args
            print("Running all algorithms for test", i)
            
            start = time.time()
            poly_value, delays, _ = run_one_network(instance, Fs, weight_matrix, nb_routes, period, mct, B, mode="POLYOPTI")
            poly_computation_time = time.time() - start
            if poly_value == -1:
                delays = None
            """start = time.time()
            greedy_value = run_one_network(instance, Fs, weight_matrix, nb_routes, period, mct, B, mode="Greedy")
            greedy_computation_time = time.time() - start"""
            greedy_computation_time = 0
            greedy_value="Not computed"
            print(f"Poly value: {poly_value}, Greedy value: {greedy_value}, instance {i}")
            algo = "LRI"
            glob_S_list = []


            # Créer une nouvelle instance de paquets et de liens pour chaque exécution
            paquets, liens = creer_paquets_depuis_instance(instance, Fs, weight_matrix, period, mct, B, delays, mode=algo)
            start = time.time()
            #glob_s_min, average_utility, fail_RL = run_learning_iterations(paquets, liens, period, nb_iterations=80000, alpha=0.001, poly_value=poly_value, mode=algo, submode="localinteraction", it_number=i, timestamp=timestamp, base_dir="./results/RL", ax=None)
            glob_s_min, average_utility, fail_RL = run_learning_phases_sequential(
                paquets, liens, period,
                nb_phases=10000,  # approx même nombre total de "coups"
                alpha=0.001,
                poly_value=poly_value,
                mode=algo,
                submode=submode,
                selection_mode=sort  # "sorted" ou "random"
            )

            RL_computation_time = time.time() - start
            glob_S_list.append(glob_s_min)

            # Écrire les résultats dans un fichier CSV spécifique au thread
            filename = f"results_{submode}_{sort}_r{nb_routes}_l{nb_levels}_p{period}_c{mct}_i{i}.csv"
            result = {
                "algo": algo,
                "run": i,
                "glob_s_min": glob_S_list,
                "poly_value": poly_value,
                "average_utility": average_utility,
                "fail_RL": fail_RL,
                "greedy_value": greedy_value,
                "poly_computation_time": poly_computation_time,
                "greedy_computation_time": greedy_computation_time,
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


if __name__ == "__main__":
    nb_tests = 100
    nb_routes = 5
    nb_levels = 2
    period = 10
    mct = 2
    B = 3

    # for nb_levels in tqdm([3], desc="Number of levels"):
    timestamp = time.strftime("%d-%m-%Y_%H-%M-%S" + str(nb_routes) + "routes_" + str(nb_levels) + "levels")
    #     for period in tqdm([10], desc="Period", leave=True):
    args_list = []
    for i in range(nb_tests):
        #instance, Fs, weight_matrix, _ = generate_instance_and_traffic(nb_routes, nb_levels, period, mct)
        data = get_instance_from_dataset(nb_routes, nb_levels, period, mct, i)
        instance, S, Fs, weight_matrix = data[4], data[5], data[6], data[7]
        args_list.append((i, instance, Fs, weight_matrix, nb_routes, period, mct, B, timestamp, nb_levels))

    # Utiliser Pool pour paralléliser les tâches et limiter le nombre de cœurs
    num_cores = max(1, multiprocessing.cpu_count())  # Limitez le nombre de cœurs à utiliser
    num_cores = min(num_cores,nb_tests)
    with Pool(processes=num_cores) as pool:
        pool.map(run_all_algos_no_fig, args_list)
        # generate_global_figure(base_dir, period, nb_tests)