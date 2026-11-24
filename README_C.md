# Bloc C : optimisation et expériences

Ce document décrit le travail du membre C : la boucle d'apprentissage, la
comparaison des taux d'apprentissage, du momentum et de la taille du réseau,
et l'étude de la convergence. Il complète le `README.md` du membre B (réseau
et rétropropagation). Le réseau lui-même (`src/network.py`) n'est pas modifié.

## Ce que fait ce bloc

Le réseau de B calcule le coût et les gradients. Mon code utilise ces
gradients pour mettre à jour les poids et les biais, puis compare plusieurs
façons d'apprendre dans des conditions identiques.

## Fichiers ajoutés

```text
src/optimisation.py          # boucle d'apprentissage (descente simple + momentum)
src/experiences.py           # lance les expériences, sauve les CSV et les courbes
test_validite.py             # vérifie que ma boucle reproduit celle de B
resultats/experiences/       # CSV et images produits par experiences.py
rapport/partie_C.tex         # ma partie du rapport
README_C.md                  # ce fichier
```

## Ce que j'utilise du code de B

| Élément de `src/network.py`               | Usage dans mon code                           |
| ----------------------------------------- | --------------------------------------------- |
| `ReseauNeurones(57, n_cache, 10, graine)` | réseau avec des poids de départ reproductibles |
| `W1`, `b1`, `W2`, `b2`                    | paramètres modifiés en place                  |
| `retropropagation(X, Y)`                  | renvoie `(gradients, perte)`                  |
| `propagation_avant(X)[3]`                 | sorties `a2`, pour mesurer coût et précision  |
| `predire(X)`                              | prédictions pour la précision de test         |
| `cout(a, y)`                              | fonction de coût                              |

Je n'utilise pas la méthode `entrainer` de B : j'ai écrit ma propre boucle
pour pouvoir la réutiliser et la comparer dans les expériences.

## Boucle d'apprentissage

Mise à jour utilisée (`src/optimisation.py`) :

```text
v     <- momentum * v + gradient
theta <- theta - eta * v
```

Avec `momentum = 0`, c'est la descente de gradient simple. La vitesse `v` est
créée une seule fois avant l'entraînement et remise à zéro au début de chaque
nouvelle expérience.

Cette formule est équivalente à celle de B (`v <- mu*v - eta*g`, `w <- w + v`)
en posant `v_B = -eta * v`.

À chaque époque :

1. les 787 morceaux sont mélangés ;
2. on les découpe en mini-lots de 64 (13 mises à jour par époque) ;
3. le coût et la précision sont mesurés à la fin de l'époque, sur tout le
   jeu d'apprentissage.

Si le coût n'est plus fini (`NaN` ou `inf`), l'entraînement s'arrête : on
considère que l'essai a divergé.

## Installation et exécution

Python 3.10+ et NumPy, comme pour B. Matplotlib est nécessaire pour tracer
les courbes (outil de visualisation, pas de machine learning). Il faut
l'ajouter à `requirements.txt`.

```bash
.venv/bin/pip install matplotlib
.venv/bin/python test_validite.py
.venv/bin/python -m src.experiences
```

Le lancement se fait depuis la racine du projet.

## Protocole expérimental

- mêmes données (celles de A) ;
- 800 époques et mini-lots de 64 ;
- même graine (42) : mêmes poids de départ et même ordre des mini-lots ;
- un seul paramètre change d'une expérience à l'autre ;
- le coût est toujours mesuré de la même façon (fin d'époque, jeu complet).

| Expérience              | Paramètre qui varie        | Valeurs                        | Sortie                             |
| ----------------------- | -------------------------- | ------------------------------ | ---------------------------------- |
| 1. Taux d'apprentissage | `eta` (momentum = 0)       | 0,001 ; 0,01 ; 0,1 ; 0,5 ; 1,0 | `exp1_taux_apprentissage.csv/.png` |
| 2. Momentum             | `momentum` (eta = 0,1)     | 0 ; 0,5 ; 0,9                  | `exp2_momentum.csv/.png`           |
| 3. Taille du réseau     | `n_cache` (eta = 0,5)      | 8 ; 16 ; 32 ; 64 ; 128         | `exp3_taille_reseau.csv/.png`      |

La précision de test est indicative : l'évaluation officielle (précision,
matrice de confusion) est la partie du membre A. Le jeu de test contient 198
morceaux, donc un morceau vaut environ 0,5 point.

## Vérification de ma boucle

- [x] `eta = 0.5`, `momentum = 0`, 800 époques, graine 42 : les résultats
      ressemblent à ceux de B (coût final 0,0308, précision d'apprentissage
      84,75 %, précision de test 72,22 %).

Cette vérification est reprise par l'expérience 1 (`eta = 0.5`) et
l'expérience 3 (`n_cache = 32`), qui donnent les mêmes valeurs.

## Résultats

### Expérience 1 : taux d'apprentissage (descente simple)

| eta   | Coût final | Préc. train (%) | Préc. test (%) | Temps (s) |
| ----: | ---------: | --------------: | -------------: | --------: |
| 0,001 | 0,1118     | 13,98           | 14,65          | 4,0       |
| 0,01  | 0,0834     | 33,80           | 33,84          | 3,7       |
| 0,1   | 0,0599     | 58,58           | 57,58          | 3,7       |
| 0,5   | 0,0308     | 84,75           | 72,22          | 4,0       |
| 1,0   | 0,0199     | 91,87           | 72,73          | 3,8       |

### Expérience 2 : momentum (eta = 0,1)

| momentum | Pas effectif | Coût final | Préc. train (%) | Préc. test (%) | Temps (s) |
| -------: | -----------: | ---------: | --------------: | -------------: | --------: |
| 0        | 0,1          | 0,0599     | 58,58           | 57,58          | 4,0       |
| 0,5      | 0,2          | 0,0481     | 72,68           | 62,63          | 4,0       |
| 0,9      | 1,0          | 0,0200     | 92,25           | 73,23          | 4,3       |

Le pas effectif vaut `eta / (1 - momentum)` quand le gradient est à peu près
constant.

### Expérience 3 : taille du réseau (eta = 0,5)

| Neurones cachés | Paramètres | Coût final | Préc. train (%) | Préc. test (%) | Temps (s) |
| --------------: | ---------: | ---------: | --------------: | -------------: | --------: |
| 8               | 554        | 0,0438     | 74,08           | 63,13          | 2,0       |
| 16              | 1098       | 0,0353     | 83,35           | 70,20          | 2,9       |
| 32              | 2186       | 0,0308     | 84,75           | 72,22          | 3,9       |
| 64              | 4362       | 0,0288     | 86,79           | 71,21          | 8,8       |
| 128             | 8714       | 0,0282     | 86,66           | 72,73          | 16,1      |

Les temps dépendent de l'ordinateur utilisé.

## Observations

- **Taux d'apprentissage** : plus `eta` est grand, plus le coût baisse vite.
  Aucune oscillation ni divergence jusqu'à `eta = 1,0`. Avec `eta = 0,001`, la
  précision d'apprentissage reste à 14 % (environ 10 % au hasard).
- **Coût et test** : de `eta = 0,5` à `eta = 1,0`, le coût baisse de 0,0308 à
  0,0199 mais la précision de test passe de 72,22 % à 72,73 % (un morceau).
  L'écart entre apprentissage et test grandit (environ 12,5 puis 19,1
  points) : début de surapprentissage.
- **Momentum** : avec `eta = 0,1`, `momentum = 0,9` donne presque le même
  résultat que la descente simple avec `eta = 1,0` (coûts 0,01997 et 0,01995).
  Dans ces essais, il agit surtout comme un pas plus grand ; aucun avantage de
  stabilité n'a été mis en évidence.
- **Taille du réseau** : de 8 à 32 neurones, le coût et la précision de test
  s'améliorent nettement. Au-delà de 32, le gain sur le coût est faible, le
  temps augmente fortement et la précision de test ne change pas de façon
  nette (1 à 3 morceaux d'écart).
- **Convergence** : à 800 époques, aucune courbe n'est stabilisée (elles
  baissent encore). On ne peut pas dire que le réseau a convergé, seulement
  que le coût diminue régulièrement.

## Limites

- Une seule graine par essai : une petite différence peut venir du hasard.
- Jeu de test de 198 morceaux : un écart de quelques morceaux n'est pas
  significatif.
- 800 époques ne suffisent pas pour voir une stabilisation du coût.
- La limite de stabilité de `eta` n'a pas été cherchée (aucun essai
  au-dessus de 1,0).
- Les mini-lots ajoutent du bruit au gradient.
- Sigmoïdes avec erreur quadratique (choix de B) ; softmax et entropie
  croisée seraient une extension possible.
- Méthodes d'ordre deux (Newton) discutées dans le rapport, mais non testées :
  la hessienne ferait 2186 x 2186 entrées.


## Lien avec le rapport

Ma partie est dans `rapport/partie_C.tex`. Les figures sont lues dans
`resultats/experiences/`.
