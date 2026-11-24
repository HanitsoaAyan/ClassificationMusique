import numpy as np

from src.network import cout


def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]


def entrainer(reseau, X, y, eta=0.1, epoques=200, taille_lot=64,
              momentum=0.0, graine=42):
    """Descente de gradient par mini-lots, avec momentum optionnel.

    v <- momentum * v + gradient
    theta <- theta - eta * v
    Avec momentum = 0, c'est la descente de gradient simple.
    """
    Y = one_hot(y)
    N = X.shape[0]
    rng = np.random.default_rng(graine)

    parametres = {"W1": reseau.W1, "b1": reseau.b1,
                  "W2": reseau.W2, "b2": reseau.b2}

    # La vitesse est créée une seule fois, avant l'entraînement
    vitesses = {}
    for nom in parametres:
        vitesses[nom] = np.zeros_like(parametres[nom])

    historique_cout = []
    historique_precision = []

    for epoque in range(epoques):
        indices = rng.permutation(N)

        for debut in range(0, N, taille_lot):
            lot = indices[debut:debut + taille_lot]
            gradients, _ = reseau.retropropagation(X[lot], Y[lot])

            for nom in parametres:
                vitesses[nom] = momentum * vitesses[nom] + gradients[nom]
                parametres[nom] -= eta * vitesses[nom]

        # Coût mesuré à la FIN de chaque époque, sur tout le jeu d'apprentissage
        a2 = reseau.propagation_avant(X)[3]
        c = cout(a2, Y)
        historique_cout.append(c)
        historique_precision.append(float(np.mean(np.argmax(a2, axis=1) == y)))

        # Si le coût n'est plus fini, on arrête (divergence)
        if not np.isfinite(c):
            break

    return historique_cout, historique_precision

    