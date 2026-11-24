"""Entraînement du réseau sur les données préparées .

Entrées  : data/X_train.npy (787, 57), data/y_train.npy (787,)
Test     : data/X_test.npy  (198, 57), data/y_test.npy  (198,)

Sorties produites :
    - data/y_pred.npy       : les 198 prédictions demandées par le membre A ;
    - data/y_pred.csv       : les mêmes prédictions, en CSV ;
    - resultats/modele.npz  : les poids et biais du réseau entraîné ;
    - resultats/historique.csv : coût et précision à chaque époque ;
    - resultats/probas_test.npy : les 10 probabilités par morceau de test.

Usage :
    .venv/bin/python src/train.py
    .venv/bin/python src/train.py --epoques 1000 --eta 0.3 --momentum 0.9
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.network import ReseauNeurones, one_hot  # noqa: E402

DATA = RACINE / "data"
RESULTATS = RACINE / "resultats"


def charger_donnees() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Charge les tableaux du membre A et vérifie leurs formes."""
    X_train = np.load(DATA / "X_train.npy")
    X_test = np.load(DATA / "X_test.npy")
    y_train = np.load(DATA / "y_train.npy")
    y_test = np.load(DATA / "y_test.npy")

    attendus = {
        "X_train": ((787, 57), X_train.shape),
        "X_test": ((198, 57), X_test.shape),
        "y_train": ((787,), y_train.shape),
        "y_test": ((198,), y_test.shape),
    }
    for nom, (forme_attendue, forme) in attendus.items():
        if forme != forme_attendue:
            raise ValueError(
                f"{nom} a la forme {forme}, attendu {forme_attendue}. "
                "Le format des données a changé : prévenir le membre A."
            )

    if np.any(np.isnan(X_train)) or np.any(np.isnan(X_test)):
        raise ValueError("Les données contiennent des NaN.")
    if set(np.unique(y_train)) != set(range(10)):
        raise ValueError("y_train doit contenir les étiquettes 0 à 9.")

    print("Données chargées :")
    print(f"  X_train {X_train.shape} | y_train {y_train.shape}")
    print(f"  X_test  {X_test.shape} | y_test  {y_test.shape}")
    return X_train, X_test, y_train, y_test


def ecrire_predictions_csv(chemin: Path, predictions: np.ndarray) -> None:
    """Écrit un CSV simple : une colonne de 198 valeurs, sans en-tête.

    Ainsi, `np.loadtxt("data/y_pred.csv", delimiter=",")` fonctionne
    directement et renvoie bien 198 nombres.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        for valeur in predictions:
            writer.writerow([int(valeur)])


def ecrire_historique_csv(chemin: Path, historique: list[dict[str, float]]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(["epoque", "cout", "precision_train"])
        for ligne in historique:
            writer.writerow(
                [
                    int(ligne["epoque"]),
                    f"{ligne['cout']:.10f}",
                    f"{ligne['precision']:.10f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement du réseau de neurones.")
    parser.add_argument("--epoques", type=int, default=800)
    parser.add_argument("--eta", type=float, default=0.5, help="pas d'apprentissage")
    parser.add_argument("--lot", type=int, default=64, help="taille des mini-lots")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--cache", type=int, default=32, help="neurones cachés")
    parser.add_argument("--graine", type=int, default=42)
    parser.add_argument("--silencieux", action="store_true")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    RESULTATS.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = charger_donnees()

    # Les 10 sorties sigmoïdes répondent chacune à « est-ce le genre k ? ».
    Y_train = one_hot(y_train, 10)

    reseau = ReseauNeurones(
        n_entrees=57,
        n_cache=args.cache,
        n_sorties=10,
        graine=args.graine,
    )

    print("\nArchitecture : 57 -> %d -> 10 (sigmoïdes)" % args.cache)
    print(
        f"Hyperparamètres : eta = {args.eta}, lot = {args.lot}, "
        f"momentum = {args.momentum}, époques = {args.epoques}, graine = {args.graine}"
    )
    print()

    historique = reseau.entrainer(
        X_train,
        Y_train,
        epoques=args.epoques,
        taille_lot=args.lot,
        eta=args.eta,
        momentum=args.momentum,
        graine=args.graine,
        verbose=not args.silencieux,
        afficher_toutes=100,
    )

    # ------------------------------------------------------------------
    # Prédictions 
    # ------------------------------------------------------------------
    probas_test = reseau.predire_proba(X_test)
    y_pred = np.argmax(probas_test, axis=1).astype(np.int64)

    if y_pred.shape != (198,):
        raise RuntimeError(f"y_pred a la forme {y_pred.shape}, attendu (198,).")
    if y_pred.min() < 0 or y_pred.max() > 9:
        raise RuntimeError("Les prédictions doivent être comprises entre 0 et 9.")

    np.save(DATA / "y_pred.npy", y_pred)
    ecrire_predictions_csv(DATA / "y_pred.csv", y_pred)

    # ------------------------------------------------------------------
   
    # ------------------------------------------------------------------
    reseau.sauvegarder(RESULTATS / "modele.npz")
    np.save(RESULTATS / "probas_test.npy", probas_test)
    ecrire_historique_csv(RESULTATS / "historique.csv", historique)

    # ------------------------------------------------------------------
    #
    # ------------------------------------------------------------------
    y_pred_train = reseau.predire(X_train)
    precision_train = float(np.mean(y_pred_train == y_train))
    precision_test = float(np.mean(y_pred == y_test))

    print("\n" + "=" * 62)
    print("BILAN")
    print("=" * 62)
    print(f"Coût final (apprentissage) : {historique[-1]['cout']:.6f}")
    print(f"Précision apprentissage    : {precision_train:.4f}")
    print(f"Précision test (indicative): {precision_test:.4f}")
    print()
    print(f"y_pred.npy : forme {y_pred.shape}, dtype {y_pred.dtype}")
    print(f"Valeurs : min = {y_pred.min()}, max = {y_pred.max()}")
    print(f"Fichier écrit : {DATA / 'y_pred.npy'}")
    print(f"Fichier écrit : {DATA / 'y_pred.csv'}")
    print(f"Modèle écrit  : {RESULTATS / 'modele.npz'}")
    print(f"Historique    : {RESULTATS / 'historique.csv'}")
    print()
    print("Rappel membre A : les 198 prédictions sont dans le même ordre")
    print("que X_test, avec l'encodage blues = 0 ... rock = 9.")


if __name__ == "__main__":
    main()
