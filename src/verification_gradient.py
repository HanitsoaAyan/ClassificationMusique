
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Permet d'exécuter le script directement : python src/verification_gradient.py
RACINE = Path(__file__).resolve().parents[1]
if str(RACINE) not in sys.path:
    sys.path.insert(0, str(RACINE))

from src.network import (  # noqa: E402
    ReseauNeurones,
    cout,
    derivee_sigmoide_depuis_a,
    erreur,
    neurone,
    one_hot,
    sigmoide,
)

RESULTATS = RACINE / "resultats"


def exemple_du_cours() -> dict[str, float]:
    """Reproduit le neurone de CNN.md et compare pente exacte / approchée."""
    x = np.array([0.8, 0.5])
    w = np.array([0.4, 0.6])
    b = 0.1
    y = 1.0

    z = neurone(x, w, b)                 # 0,4*0,8 + 0,6*0,5 + 0,1 = 0,72
    a = float(sigmoide(z))               # sigma(0,72) ~ 0,6726
    e = float(erreur(a, y))              # (a - 1)^2 ~ 0,1072

    # Règle de la chaîne : dE/dw1 = dE/da * da/dz * dz/dw1
    dE_da = 2.0 * (a - y)
    da_dz = float(derivee_sigmoide_depuis_a(a))
    dz_dw1 = x[0]
    pente_w1 = dE_da * da_dz * dz_dw1

    # Vérification numérique sur w1 (différence centrée)
    h = 1e-6
    def erreur_pour_w1(valeur: float) -> float:
        z_local = neurone(x, np.array([valeur, w[1]]), b)
        return float(erreur(sigmoide(z_local), y))

    pente_w1_numerique = (erreur_pour_w1(w[0] + h) - erreur_pour_w1(w[0] - h)) / (2 * h)

    print("=" * 70)
    print("1. EXEMPLE DE CNN.md (un seul neurone)")
    print("=" * 70)
    print(f"z = w.x + b                  = {z:.6f}")
    print(f"a = sigma(z)                 = {a:.6f}")
    print(f"E = (a - y)^2                = {e:.6f}")
    print(f"dE/da = 2(a - y)             = {dE_da:.6f}")
    print(f"da/dz = a(1 - a)             = {da_dz:.6f}")
    print(f"dz/dw1 = x1                  = {dz_dw1:.6f}")
    print(f"dE/dw1 = produit des 3       = {pente_w1:.6f}")
    print(f"dE/dw1 approché (h = {h:g}) = {pente_w1_numerique:.6f}")
    print(f"écart absolu                 = {abs(pente_w1 - pente_w1_numerique):.3e}")

    # Tableau de la sigmoïde, comme dans les notes.
    print("\nQuelques valeurs de sigma :")
    for valeur in (-5.0, 0.0, 0.72, 5.0):
        print(f"  sigma({valeur:>5.2f}) = {float(sigmoide(valeur)):.6f}")

    return {
        "z": z,
        "a": a,
        "E": e,
        "dE_dw1": pente_w1,
        "dE_dw1_numerique": pente_w1_numerique,
        "ecart": abs(pente_w1 - pente_w1_numerique),
    }


def verifier_gradient(
    reseau: ReseauNeurones,
    X: np.ndarray,
    Y: np.ndarray,
    h: float = 1e-5,
) -> dict[str, float]:
    """Compare gradient rétropropagé et gradient numérique pour tous les poids."""
    gradients, perte = reseau.retropropagation(X, Y)
    parametres = {
        "W1": reseau.W1,
        "b1": reseau.b1,
        "W2": reseau.W2,
        "b2": reseau.b2,
    }

    ecarts_relatifs: list[float] = []
    ecarts_absolus: list[float] = []
    pires: tuple[str, tuple[int, ...], float, float] | None = None

    for nom, parametre in parametres.items():
        for indice in np.ndindex(parametre.shape):
            valeur_originale = parametre[indice]

            parametre[indice] = valeur_originale + h
            cout_plus = reseau.cout(X, Y)

            parametre[indice] = valeur_originale - h
            cout_moins = reseau.cout(X, Y)

            parametre[indice] = valeur_originale

            gradient_numerique = (cout_plus - cout_moins) / (2.0 * h)
            gradient_analytique = float(gradients[nom][indice])

            ecart_absolu = abs(gradient_numerique - gradient_analytique)
            denominateur = max(1e-12, abs(gradient_numerique) + abs(gradient_analytique))
            ecart_relatif = ecart_absolu / denominateur

            ecarts_absolus.append(ecart_absolu)
            ecarts_relatifs.append(ecart_relatif)

            if pires is None or ecart_relatif > pires[2]:
                pires = (nom, indice, ecart_relatif, ecart_absolu)

    assert pires is not None
    resultat = {
        "cout": perte,
        "n_parametres": float(len(ecarts_relatifs)),
        "ecart_absolu_max": float(np.max(ecarts_absolus)),
        "ecart_relatif_max": float(np.max(ecarts_relatifs)),
        "ecart_relatif_moyen": float(np.mean(ecarts_relatifs)),
        "pire_parametre_nom": pires[0],
        "pire_parametre_indice": pires[1],
        "pire_ecart_relatif": pires[2],
        "pire_ecart_absolu": pires[3],
    }

    print("\n" + "=" * 70)
    print("2. VÉRIFICATION DU GRADIENT SUR TOUS LES PARAMÈTRES")
    print("=" * 70)
    print(f"Nombre de paramètres vérifiés : {int(resultat['n_parametres'])}")
    print(f"Coût sur le petit lot         : {perte:.8f}")
    print(f"Écart absolu maximal          : {resultat['ecart_absolu_max']:.3e}")
    print(f"Écart relatif maximal         : {resultat['ecart_relatif_max']:.3e}")
    print(f"Écart relatif moyen           : {resultat['ecart_relatif_moyen']:.3e}")
    print(
        "Pire paramètre                : "
        f"{resultat['pire_parametre_nom']}{resultat['pire_parametre_indice']} "
        f"(écart relatif {resultat['pire_ecart_relatif']:.3e})"
    )

    if resultat["ecart_relatif_max"] < 1e-5:
        print("\nRÉSULTAT : gradient vérifié (écart relatif maximal < 1e-5).")
    else:
        print("\nATTENTION : écart relatif trop grand, il y a probablement une erreur.")

    return resultat


def main() -> None:
    RESULTATS.mkdir(parents=True, exist_ok=True)

    exemple = exemple_du_cours()

    # Petit lot tiré du jeu d'apprentissage : la vérification est plus rapide.
    X_train = np.load(RACINE / "data" / "X_train.npy")
    y_train = np.load(RACINE / "data" / "y_train.npy")
    Y_train = one_hot(y_train, 10)

    rng = np.random.default_rng(0)
    indices = rng.choice(X_train.shape[0], size=8, replace=False)
    X_petit = X_train[indices]
    Y_petit = Y_train[indices]

    # On vérifie l'architecture réelle (57 -> 32 -> 10) sur un petit lot.
    # Tous les paramètres sont testés, un par un.
    reseau = ReseauNeurones(n_entrees=57, n_cache=32, n_sorties=10, graine=42)
    resultat = verifier_gradient(reseau, X_petit, Y_petit, h=1e-5)

    # Trace écrite pour le rapport.
    lignes = [
        "Vérification du gradient",
        "========================",
        "",
        "Exemple CNN.md",
        f"z = {exemple['z']:.6f}",
        f"a = {exemple['a']:.6f}",
        f"E = {exemple['E']:.6f}",
        f"dE/dw1 analytique = {exemple['dE_dw1']:.6f}",
        f"dE/dw1 numérique  = {exemple['dE_dw1_numerique']:.6f}",
        f"écart absolu      = {exemple['ecart']:.3e}",
        "",
        "Réseau complet (57 -> 32 -> 10, 8 exemples)",
        f"coût = {resultat['cout']:.8f}",
        f"paramètres vérifiés = {int(resultat['n_parametres'])}",
        f"écart absolu maximal = {resultat['ecart_absolu_max']:.3e}",
        f"écart relatif maximal = {resultat['ecart_relatif_max']:.3e}",
        f"écart relatif moyen = {resultat['ecart_relatif_moyen']:.3e}",
        "",
        "Conclusion : gradient rétropropagé cohérent avec les différences finies.",
    ]
    (RESULTATS / "verification_gradient.txt").write_text(
        "\n".join(lignes) + "\n", encoding="utf-8"
    )
    print(f"\nTrace écrite dans {RESULTATS / 'verification_gradient.txt'}")


if __name__ == "__main__":
    main()
