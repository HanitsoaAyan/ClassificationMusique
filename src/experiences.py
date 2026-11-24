import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.network import ReseauNeurones
from src.optimisation import entrainer

DOSSIER = "resultats/experiences"
GRAINE = 42
EPOQUES = 800

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")


def creer_reseau(cache):
    return ReseauNeurones(57, cache, 10, graine=GRAINE)


def lancer(nom, eta, momentum, cache=32):
    reseau = creer_reseau(cache)
    debut = time.perf_counter()
    couts, precisions = entrainer(reseau, X_train, y_train, eta=eta,
                                  epoques=EPOQUES, momentum=momentum,
                                  graine=GRAINE)
    duree = time.perf_counter() - debut
    precision_test = float(np.mean(reseau.predire(X_test) == y_test))
    return {"nom": nom, "eta": eta, "momentum": momentum, "cache": cache,
            "cout_final": couts[-1], "precision_train": precisions[-1],
            "precision_test": precision_test, "duree_s": duree,
            "couts": couts}


def sauver(resultats, fichier):
    os.makedirs(DOSSIER, exist_ok=True)
    with open(os.path.join(DOSSIER, fichier + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nom", "eta", "momentum", "cache", "cout_final",
                    "precision_train", "precision_test", "duree_s"])
        for r in resultats:
            w.writerow([r["nom"], r["eta"], r["momentum"], r["cache"],
                        r["cout_final"], r["precision_train"],
                        r["precision_test"], r["duree_s"]])

    plt.figure()
    for r in resultats:
        plt.plot(r["couts"], label=r["nom"])
    plt.xlabel("Époque")
    plt.ylabel("Coût")
    plt.title(fichier)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(DOSSIER, fichier + ".png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    # 1. Taux d'apprentissage (descente simple)
    res = []
    for eta in [0.001, 0.01, 0.1, 0.5, 1.0]:
        res.append(lancer("eta=" + str(eta), eta, 0.0))
    sauver(res, "exp1_taux_apprentissage")

    # 2. Descente simple vs momentum (même eta)
    res = []
    res.append(lancer("simple eta=0.1", 0.1, 0.0))
    res.append(lancer("momentum 0.5", 0.1, 0.5))
    res.append(lancer("momentum 0.9", 0.1, 0.9))
    sauver(res, "exp2_momentum")

    # 3. Taille du réseau
    res = []
    for cache in [8, 16, 32, 64, 128]:
        res.append(lancer("cache=" + str(cache), 0.5, 0.0, cache))
    sauver(res, "exp3_taille_reseau")

    # 4. Limite de stabilité (descente simple)
    res = []
    for eta in [2.0, 5.0, 10.0, 20.0]:
        res.append(lancer("eta=" + str(eta), eta, 0.0))
    sauver(res, "exp4_grands_taux")