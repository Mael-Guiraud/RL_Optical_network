import pandas as pd
import numpy as np
import os

# Chemin du dossier contenant les résultats
dossier = '/users/mael/NLO-PFE/results/RL/09-10-2025_17-03-165routes_2levels/'

# Liste pour stocker les résultats
resultats = []

# Parcours des fichiers de i0 à i99
for i in range(100):
    nom_fichier = f'results_r5_l2_p10_c2_i{i}.csv'
    chemin_complet = os.path.join(dossier, nom_fichier)
    
    try:
        df = pd.read_csv(chemin_complet)
        valeurs = eval(df['glob_s_min'][0], {"np": np})
        valeurs = list(valeurs)

        val_min = min(valeurs)
        val_max = max(valeurs)
        val_moyenne = sum(valeurs) / len(valeurs)

        resultats.append({
            'instance': f'i{i}',
            'min': val_min,
            'max': val_max,
            'moyenne': val_moyenne
        })

    except Exception as e:
        print(f"Erreur sur {nom_fichier} : {e}")

# Convertir en DataFrame
df_resultats = pd.DataFrame(resultats)

# Sauvegarder dans un fichier CSV
df_resultats.to_csv('/users/mael/NLO-PFE/stats_glob_s_min.csv', index=False)

print("Fichier stats_glob_s_min.csv généré avec succès.")
