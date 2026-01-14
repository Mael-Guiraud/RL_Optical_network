import numpy as np
from collections import Counter
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