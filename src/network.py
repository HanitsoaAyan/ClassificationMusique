"""



    z = w . x + b              (somme pondérée d'un neurone)
    a = sigma(z)               (fonction d'activation sigmoïde)
    E = (a - y)^2              (erreur quadratique)

Pour la classification en 10 genres, on utilise :
    - 57 entrées (les caractéristiques d'un morceau) ;
    - une couche cachée de neurones sigmoïdes ;
    - 10 sorties sigmoïdes (une par genre, encodage one-hot).

L'apprentissage se fait par descente de gradient .
.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Fonctions mathématiques de base
# ---------------------------------------------------------------------------

def sigmoide(z: np.ndarray) -> np.ndarray:
    """Calcule sigma(z) = 1 / (1 + exp(-z)) de façon numériquement stable.

    La forme naïve 1 / (1 + exp(-z)) déborde pour les grands z négatifs
    (exp(-z) devient infini). On traite séparément les z positifs et négatifs
    pour éviter cet overflow sans changer le résultat mathématique.
    """
    z = np.asarray(z, dtype=np.float64)
    resultat = np.empty_like(z)

    positifs = z >= 0
    resultat[positifs] = 1.0 / (1.0 + np.exp(-z[positifs]))

    # Pour z < 0 : sigma(z) = exp(z) / (1 + exp(z))
    exp_z = np.exp(z[~positifs])
    resultat[~positifs] = exp_z / (1.0 + exp_z)

    return resultat


def derivee_sigmoide_depuis_a(a: np.ndarray) -> np.ndarray:
    """Pente de la sigmoïde exprimée à partir de sa sortie a.

    La dérivée de sigma est sigma'(z) = sigma(z) * (1 - sigma(z)) = a(1 - a).
    C'est cette formule qui est utilisée dans la rétropropagation.
    """
    a = np.asarray(a, dtype=np.float64)
    return a * (1.0 - a)


def erreur(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Erreur quadratique E = (a - y)^2, composante par composante."""
    a = np.asarray(a, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (a - y) ** 2


def cout(a: np.ndarray, y: np.ndarray) -> float:
    """Erreur moyenne sur toutes les composantes (échantillons x sorties).

    Pour un seul neurone et un seul exemple, on retrouve exactement
    E = (a - y)^2. La moyenne permet d'obtenir une valeur indépendante
    de la taille du lot et de la taille de la couche de sortie.
    """
    return float(np.mean(erreur(a, y)))


def neurone(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """Somme pondérée d'un neurone : z = w . x + b.

    
    """
    return float(np.dot(w, x) + b)


def one_hot(y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    
    y = np.asarray(y, dtype=int)
    return np.eye(n_classes, dtype=np.float64)[y]


# ---------------------------------------------------------------------------
# Réseau de neurones
# ---------------------------------------------------------------------------

class ReseauNeurones:
    """Réseau 57 -> n_cache -> 10 avec activations sigmoïdes.

    Les poids sont stockés sous forme de matrices :
        W1 : (n_entrees, n_cache)
        b1 : (n_cache,)
        W2 : (n_cache, n_sorties)
        b2 : (n_sorties,)

    Pour un lot X de forme (N, n_entrees), chaque ligne est un exemple x^T.
    Le calcul matriciel X @ W1 + b1 réalise donc, pour chaque exemple,
    z = W^T x + b.
    """

    def __init__(
        self,
        n_entrees: int = 57,
        n_cache: int = 32,
        n_sorties: int = 10,
        graine: int = 42,
    ) -> None:
        self.n_entrees = int(n_entrees)
        self.n_cache = int(n_cache)
        self.n_sorties = int(n_sorties)
        self.graine = int(graine)

        rng = np.random.default_rng(self.graine)

        
        self.W1 = rng.normal(
            0.0,
            np.sqrt(2.0 / (self.n_entrees + self.n_cache)),
            size=(self.n_entrees, self.n_cache),
        )
        self.b1 = np.zeros(self.n_cache, dtype=np.float64)
        self.W2 = rng.normal(
            0.0,
            np.sqrt(2.0 / (self.n_cache + self.n_sorties)),
            size=(self.n_cache, self.n_sorties),
        )
        self.b2 = np.zeros(self.n_sorties, dtype=np.float64)

   
    @staticmethod
    def activation(z: np.ndarray) -> np.ndarray:
        """a = sigma(z)."""
        return sigmoide(z)

    # -- Propagation avant --------------------------------------------------
    def propagation_avant(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calcule les sorties de toutes les couches.

        Étapes :
            z1 = X W1 + b1
            a1 = sigma(z1)
            z2 = a1 W2 + b2
            a2 = sigma(z2)

        Retourne (z1, a1, z2, a2). 
        """
        X = np.asarray(X, dtype=np.float64)
        z1 = X @ self.W1 + self.b1
        a1 = sigmoide(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoide(z2)
        return z1, a1, z2, a2

    def predire_proba(self, X: np.ndarray) -> np.ndarray:
        """Renvoie les 10 probabilités (sigmoïdes) pour chaque exemple."""
        return self.propagation_avant(X)[3]

    def predire(self, X: np.ndarray) -> np.ndarray:
        """Renvoie le genre le plus probable (indice 0..9)."""
        return np.argmax(self.predire_proba(X), axis=1)

    def cout(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Erreur quadratique moyenne du réseau sur un ensemble."""
        return cout(self.predire_proba(X), Y)

    # -- Rétropropagation ---------------------------------------------------
    def retropropagation(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float]:
        """Calcule le gradient de l'erreur par rapport aux poids et biais.

        On applique la règle de la chaîne couche par couche.

        Pour la sortie, avec E = moyenne((a2 - Y)^2) :

            dE/dz2 = (2 / (N * K)) * (a2 - Y) * a2 * (1 - a2)

        Puis :
            dE/dW2 = a1^T @ dE/dz2
            dE/db2 = somme(dE/dz2)

            dE/da1 = dE/dz2 @ W2^T
            dE/dz1 = dE/da1 * a1 * (1 - a1)
            dE/dW1 = X^T @ dE/dz1
            dE/db1 = somme(dE/dz1)

        Retourne (gradients, valeur_du_cout).
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X doit avoir la forme (N, n_entrees).")
        if Y.ndim != 2:
            raise ValueError("Y doit avoir la forme (N, n_sorties).")

        N = X.shape[0]
        K = self.n_sorties

        z1, a1, z2, a2 = self.propagation_avant(X)
        perte = cout(a2, Y)

        # Lien a2 -> E : dE/da2 = 2(a2 - Y) / (N*K)
        # Lien z2 -> a2 : da2/dz2 = a2(1 - a2)
        delta2 = (2.0 / (N * K)) * (a2 - Y) * derivee_sigmoide_depuis_a(a2)

        dW2 = a1.T @ delta2
        db2 = np.sum(delta2, axis=0)

        # Lien a1 -> z2 : da2/dz2 @ W2^T, puis z1 -> a1 : a1(1 - a1)
        delta1 = (delta2 @ self.W2.T) * derivee_sigmoide_depuis_a(a1)

        dW1 = X.T @ delta1
        db1 = np.sum(delta1, axis=0)

        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return gradients, perte

    # -- Apprentissage ------------------------------------------------------
    def entrainer(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epoques: int = 800,
        taille_lot: int = 64,
        eta: float = 0.5,
        momentum: float = 0.0,
        graine: int = 42,
        verbose: bool = True,
        afficher_toutes: int = 100,
    ) -> list[dict[str, float]]:
        """Entraîne le réseau par descente de gradient sur mini-lots.

        Mise à jour de base (CNN.md) :
            w <- w - eta * dE/dw

        Avec momentum (optionnel, utile pour aller plus vite) :
            v <- momentum * v - eta * dE/dw
            w <- w + v

        momentum = 0 redonne exactement la descente de gradient simple.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X et Y doivent avoir le même nombre d'exemples.")

        rng = np.random.default_rng(graine)
        N = X.shape[0]

        parametres = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        vitesses = {nom: np.zeros_like(p) for nom, p in parametres.items()}

        historique: list[dict[str, float]] = []

        for epoque in range(1, epoques + 1):
            indices = rng.permutation(N)

            for debut in range(0, N, taille_lot):
                lot = indices[debut : debut + taille_lot]
                gradients, _ = self.retropropagation(X[lot], Y[lot])

                for nom, p in parametres.items():
                    # v <- momentum * v - eta * gradient
                    vitesses[nom] = momentum * vitesses[nom] - eta * gradients[nom]
                    # w <- w + v
                    p += vitesses[nom]

            # Mesures sur tout l'ensemble d'apprentissage, une fois par époque.
            a_train = self.predire_proba(X)
            perte = cout(a_train, Y)
            predictions = np.argmax(a_train, axis=1)
            etiquettes = np.argmax(Y, axis=1)
            precision = float(np.mean(predictions == etiquettes))

            historique.append(
                {"epoque": float(epoque), "cout": perte, "precision": precision}
            )

            if verbose and (epoque == 1 or epoque % afficher_toutes == 0):
                print(
                    f"époque {epoque:4d} | coût = {perte:.6f} "
                    f"| précision entraînement = {precision:.3f}"
                )

        return historique

    # -- Sauvegarde / chargement -------------------------------------------
    def sauvegarder(self, chemin: str | Path) -> None:
        """Sauvegarde les paramètres dans un fichier .npz."""
        chemin = Path(chemin)
        chemin.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            chemin,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            n_entrees=self.n_entrees,
            n_cache=self.n_cache,
            n_sorties=self.n_sorties,
        )

    @classmethod
    def charger(cls, chemin: str | Path) -> "ReseauNeurones":
        """Recharge un réseau sauvegardé avec sauvegarder()."""
        donnees = np.load(Path(chemin), allow_pickle=False)
        reseau = cls(
            n_entrees=int(donnees["n_entrees"]),
            n_cache=int(donnees["n_cache"]),
            n_sorties=int(donnees["n_sorties"]),
        )
        reseau.W1 = donnees["W1"].copy()
        reseau.b1 = donnees["b1"].copy()
        reseau.W2 = donnees["W2"].copy()
        reseau.b2 = donnees["b2"].copy()
        return reseau
