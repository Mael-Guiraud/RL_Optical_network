import numpy as np
from functools import lru_cache
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
    
    def compute_bound_delay(self):
        """
        Calcule une borne supérieure du délai total pour CE paquet.
        Délai max = nb_liens * (P + mct - 1)
        """
        return self.nb_liens * (self.P + self.mct - 1)

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
