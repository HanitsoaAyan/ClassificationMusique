# Classification de musique — Réseau de neurones et rétropropagation (membre B)

Ce dépôt contient le bloc **« réseau et rétropropagation »** du projet de
classification de genres musicaux. Le réseau est codé **from scratch avec
NumPy uniquement** : aucune bibliothèque de machine learning, aucun
`sklearn`, aucun `pytorch`.

Le membre A a préparé les données. Le membre B (ce code) construit le réseau,
calcule l'erreur, fait la rétropropagation et vérifie le gradient. Une boucle
d'apprentissage simple est fournie pour entraîner le modèle et produire les
prédictions demandées par le membre A.

---

- un fichier **`data/y_pred.npy`** contenant **198 prédictions** ;
- dans le **même ordre** que `X_test` ;
- avec l'encodage **blues = 0, classical = 1, country = 2, disco = 3,
  hiphop = 4, jazz = 5, metal = 6, pop = 7, reggae = 8, rock = 9**.

C'est exactement ce que produit `src/train.py` :

```text
data/y_pred.npy   -> shape (198,), dtype int64, valeurs entre 0 et 9
data/y_pred.csv   -> même chose en CSV (une colonne de 198 valeurs, sans en-tête)
```

Les formes des données n'ont **pas** été modifiées :

| Fichier            |       Forme | Rôle                                |
| ------------------ | ----------: | ----------------------------------- |
| `data/X_train.npy` | `(787, 57)` | entrées d'apprentissage             |
| `data/X_test.npy`  | `(198, 57)` | entrées de test                     |
| `data/y_train.npy` |    `(787,)` | étiquettes d'apprentissage          |
| `data/y_test.npy`  |    `(198,)` | étiquettes de test                  |
| `data/y_pred.npy`  |    `(198,)` | prédictions produites par le réseau |

> L'évaluation officielle (précision, matrice de confusion, analyse des
> erreurs) reste la partie du membre A. Le réseau ne fait que fournir ses
> prédictions.

---

## 2. Arborescence

```text
ClassificationMusique/
├── data/
│   ├── X_train.npy              # fourni par A
│   ├── X_test.npy               # fourni par A
│   ├── y_train.npy              # fourni par A
│   ├── y_test.npy               # fourni par A
│   ├── y_pred.npy               # PRODUIT PAR B : les 198 prédictions
│   └── y_pred.csv               # PRODUIT PAR B : même chose en CSV
├── resultats/
│   ├── modele.npz               # poids et biais du réseau entraîné
│   ├── historique.csv           # coût et précision à chaque époque
│   ├── probas_test.npy          # 10 scores sigmoïdes par morceau de test
│   └── verification_gradient.txt
├── src/
│   ├── __init__.py
│   ├── network.py               # neurone, sigmoïde, erreur, réseau, rétropropagation
│   ├── train.py                 # entraînement + génération de y_pred.npy
│   └── verification_gradient.py # vérification numérique du gradient
├── requirements.txt             # numpy uniquement
├── CNN.md                       # notes de cours / notations du membre B
├── Data.odt                     # consignes du membre A
├── Note_de_cadrage_projet_reseau_de_neurones.pdf
├── music.ipynb                  # préparation des données (membre A)
└── README.md
```

---

## 3. Installation et exécution

Le projet n'utilise que NumPy et demande Python 3.10+ (le `.venv` fourni
utilise Python 3.12). Si l'environnement `.venv` existe déjà :

```bash
# 1. Vérifier le gradient (à faire en premier)
.venv/bin/python src/verification_gradient.py

# 2. Entraîner le réseau et produire les prédictions
.venv/bin/python src/train.py
```

Si besoin de recréer l'environnement :

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Options disponibles pour l'entraînement :

```bash
.venv/bin/python src/train.py \
    --epoques 800 \
    --eta 0.5 \
    --lot 64 \
    --momentum 0.0 \
    --cache 32 \
    --graine 42
```

| Option       | Valeur par défaut | Signification                                 |
| ------------ | ----------------: | --------------------------------------------- |
| `--epoques`  |             `800` | nombre de passages sur le jeu d'apprentissage |
| `--eta`      |             `0.5` | pas d'apprentissage $\eta$                    |
| `--lot`      |              `64` | taille des mini-lots                          |
| `--momentum` |             `0.0` | momentum (0 = descente de gradient simple)    |
| `--cache`    |              `32` | nombre de neurones de la couche cachée        |
| `--graine`   |              `42` | graine aléatoire, pour être reproductible     |

Le script est déterministe : avec la même graine et les mêmes options, on
retrouve les mêmes prédictions (aux erreurs d'arrondi flottantes près).

---

## 4. Les données et l'encodage

1. nettoyé le dataset GTZAN (`music_propre.csv`) ;
2. gardé **57 caractéristiques** par morceau :
   - 8 mesures du son, en moyenne et en variance : 16 nombres ;
   - le tempo : 1 nombre ;
   - 20 coefficients MFCC, en moyenne et en variance : 40 nombres ;
3. normalisé les données avec la moyenne et l'écart-type **du train
   uniquement** ;
4. découpé en 787 morceaux d'apprentissage et 198 morceaux de test ;
5. encodé les 10 genres de 0 à 9.

Le réseau reçoit donc une matrice $X$ de forme $(N, 57)$. Chaque ligne est un
morceau, chaque colonne est une caractéristique.

Pour la sortie, on transforme l'étiquette $y \in \{0,\dots,9\}$ en **vecteur
one-hot** de taille 10 :

```python
def one_hot(y, n_classes=10):
    return np.eye(n_classes)[y]
```

Exemple : le genre 3 (disco) devient

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

La $k$-ième sortie du réseau répond à la question :

> « Ce morceau est-il du genre $k$ ? »

C'est pour cela qu'on utilise **10 sorties sigmoïdes indépendantes** : chaque
sortie est une réponse oui/non, entre 0 et 1. Le genre prédit est celui dont
la sortie est la plus grande :

```python
y_pred = np.argmax(probas, axis=1)
```

> Remarque : avec des sigmoïdes indépendantes, les 10 scores ne sont pas
> forcés à sommer à 1. Ce sont des scores de confiance. Le genre retenu est
> le plus grand. Une variante avec softmax + entropie croisée est discutée
> en section 13.

---

## 5. Étape 1 — Le neurone

Un neurone calcule une **somme pondérée** de ses entrées, puis ajoute un
biais :

$$
z = w \cdot x + b = \sum_{i=1}^{57} w_i x_i + b
$$

où :

- $x$ est le vecteur des entrées (les 57 caractéristiques) ;
- $w$ est le vecteur des poids ;
- $b$ est le biais ;
- $z$ est le résultat brut, avant activation.

Dans le code, la fonction de base est :

```python
def neurone(x, w, b):
    return float(np.dot(w, x) + b)
```

Avec l'exemple de `CNN.md` :

$$
z = 0{,}8 \times 0{,}4 + 0{,}5 \times 0{,}6 + 0{,}1 = 0{,}72
$$

### Version vectorisée

Au lieu de faire une boucle sur les 787 morceaux et les 32 neurones cachés,
on utilise des matrices. Si $X$ a la forme $(N, 57)$ et $W_1$ la forme
$(57, 32)$, alors :

$$
Z_1 = X W_1 + b_1
$$

donne directement les $N \times 32$ valeurs de $z$ pour tous les morceaux et
tous les neurones cachés.

---

## 6. Étape 2 — La fonction d'activation : sigmoïde

La sigmoïde écrase $z$ dans l'intervalle $]0, 1[$ :

$$
a = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Propriétés :

- $z$ très grand $\Rightarrow a \approx 1$ (« oui, très probable ») ;
- $z$ très négatif $\Rightarrow a \approx 0$ (« non, très peu probable ») ;
- $z = 0 \Rightarrow a = 0{,}5$ (« incertain »).

Quelques valeurs :

|      $z$ |  $\sigma(z)$ |
| -------: | -----------: |
|     $-5$ | $0{,}006693$ |
|      $0$ | $0{,}500000$ |
| $0{,}72$ | $0{,}672607$ |
|      $5$ | $0{,}993307$ |

### Dérivée de la sigmoïde

C'est la formule centrale de la rétropropagation :

$$
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr) = a(1-a)
$$

Démonstration rapide :

$$
\sigma(z) = (1+e^{-z})^{-1}
$$

$$
\sigma'(z) = -(1+e^{-z})^{-2} \cdot (-e^{-z})
= \frac{e^{-z}}{(1+e^{-z})^2}
$$

Or

$$
\sigma(z)\bigl(1-\sigma(z)\bigr)
= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}}
= \frac{e^{-z}}{(1+e^{-z})^2}
= \sigma'(z).
$$

### Implémentation numériquement stable

La formule naïve `1 / (1 + np.exp(-z))` déborde pour les grands $z$ négatifs
car `np.exp(-z)` devient infini. On sépare les cas :

```python
def sigmoide(z):
    z = np.asarray(z, dtype=np.float64)
    resultat = np.empty_like(z)

    positifs = z >= 0
    resultat[positifs] = 1.0 / (1.0 + np.exp(-z[positifs]))

    exp_z = np.exp(z[~positifs])
    resultat[~positifs] = exp_z / (1.0 + exp_z)

    return resultat
```

Pour $z < 0$, on utilise la forme équivalente :

$$
\sigma(z) = \frac{e^z}{1+e^z}
$$

qui ne calcule jamais $e^{-z}$ avec un grand exposant positif.

Dans `CNN.md`, l'exemple donne :

$$
\sigma(0{,}72) \approx 0{,}672607
$$

---

## 7. Étape 3 — L'erreur

Pour un neurone et un exemple, l'erreur quadratique est :

$$
E = (a - y)^2
$$

où $y$ est la bonne réponse (0 ou 1) et $a$ la réponse du neurone.

Avec l'exemple de `CNN.md`, pour un morceau classique ($y = 1$) :

$$
a = 0{,}672607
\quad\Rightarrow\quad
E = (0{,}672607 - 1)^2 = 0{,}107186
$$

La dérivée de l'erreur par rapport à la sortie est :

$$
\frac{\partial E}{\partial a} = 2(a-y)
$$

Pour l'exemple :

$$
2(0{,}672607 - 1) = -0{,}654786
$$

Le signe négatif signifie : « si $a$ augmente, $E$ diminue ». C'est normal,
puisque $a$ est encore en dessous de la bonne réponse 1.

### Pour les 10 sorties

Le réseau a 10 sorties. On moyenne l'erreur sur les $N$ exemples du lot et
les $K = 10$ sorties :

$$
E = \frac{1}{NK} \sum_{n=1}^{N} \sum_{k=1}^{K} (a_{n,k} - y_{n,k})^2
$$

Le facteur $\frac{1}{NK}$ ne change pas la direction du gradient, seulement
son échelle. Il rend le pas d'apprentissage $\eta$ indépendant de la taille
du lot et du nombre de classes.

Code :

```python
def erreur(a, y):
    return (a - y) ** 2

def cout(a, y):
    return float(np.mean(erreur(a, y)))
```

---

## 8. Étape 4 — Architecture et propagation avant

L'architecture choisie est :

$$
57 \longrightarrow 32 \longrightarrow 10
$$

- **57 entrées** : les caractéristiques d'un morceau ;
- **32 neurones cachés** sigmoïdes ;
- **10 sorties** sigmoïdes, une par genre.

### Équations de la propagation avant

Pour un lot $X$ de forme $(N, 57)$ :

$$
Z_1 = X W_1 + b_1
$$

$$
A_1 = \sigma(Z_1)
$$

$$
Z_2 = A_1 W_2 + b_2
$$

$$
A_2 = \sigma(Z_2)
$$

Dimensions :

| Objet      | Forme      | Signification                          |
| ---------- | ---------- | -------------------------------------- |
| $X$        | $(N, 57)$  | entrées                                |
| $W_1$      | $(57, 32)$ | poids entrée → caché                   |
| $b_1$      | $(32,)$    | biais cachés                           |
| $Z_1, A_1$ | $(N, 32)$  | pré-activations et activations cachées |
| $W_2$      | $(32, 10)$ | poids caché → sortie                   |
| $b_2$      | $(10,)$    | biais de sortie                        |
| $Z_2, A_2$ | $(N, 10)$  | scores des 10 genres                   |

Code :

```python
def propagation_avant(self, X):
    z1 = X @ self.W1 + self.b1
    a1 = sigmoide(z1)
    z2 = a1 @ self.W2 + self.b2
    a2 = sigmoide(z2)
    return z1, a1, z2, a2
```

Les valeurs intermédiaires `z1`, `a1`, `z2`, `a2` sont conservées : la
rétropropagation en a besoin.

### Initialisation des poids

Les poids sont initialisés avec une loi normale centrée, à l'échelle de
Xavier/Glorot :

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{entrée}} + n_{\text{sortie}}}}\right)
$$

et les biais à zéro. Si les poids étaient trop grands, les sigmoïdes
satureraient dès le début et le gradient serait presque nul. Si les poids
étaient tous nuls, tous les neurones apprendraient la même chose.

---

## 9. Étape 5 — Rétropropagation

La rétropropagation est la **règle de la chaîne** appliquée couche par
couche. Le graphe de calcul est :

$$
w \longrightarrow z \longrightarrow a \longrightarrow E
$$

### Cas d'un seul neurone

Pour un poids $w_i$ :

$$
\frac{\partial E}{\partial w_i}
=
\frac{\partial E}{\partial a}
\cdot
\frac{\partial a}{\partial z}
\cdot
\frac{\partial z}{\partial w_i}
$$

Avec :

$$
\frac{\partial E}{\partial a} = 2(a-y),
\qquad
\frac{\partial a}{\partial z} = a(1-a),
\qquad
\frac{\partial z}{\partial w_i} = x_i
$$

donc :

$$
\boxed{
\frac{\partial E}{\partial w_i}
=
2(a-y)\,a(1-a)\,x_i
}
$$

et pour le biais :

$$
\boxed{
\frac{\partial E}{\partial b}
=
2(a-y)\,a(1-a)
}
$$

Vérification numérique avec l'exemple de `CNN.md` :

$$
2 \times (0{,}672607 - 1) \times 0{,}220207 \times 0{,}8
= -0{,}115351
$$

C'est bien la valeur annoncée dans les notes.

### Cas du réseau à deux couches

On note $\delta$ les gradients par rapport aux pré-activations $z$.

Pour la couche de sortie, avec la moyenne sur $N$ exemples et $K$ sorties :

$$
\delta_2 = \frac{2}{NK}(A_2 - Y) \odot A_2 \odot (1 - A_2)
$$

où $\odot$ est le produit terme à terme.

Puis :

$$
\frac{\partial E}{\partial W_2} = A_1^T \delta_2
\qquad
\frac{\partial E}{\partial b_2} = \sum_{n=1}^{N} \delta_2^{(n)}
$$

On propage vers la couche cachée :

$$
\delta_1
=
\left(\delta_2 W_2^T\right) \odot A_1 \odot (1 - A_1)
$$

Enfin :

$$
\frac{\partial E}{\partial W_1} = X^T \delta_1
\qquad
\frac{\partial E}{\partial b_1} = \sum_{n=1}^{N} \delta_1^{(n)}
$$

Code correspondant :

```python
delta2 = (2.0 / (N * K)) * (a2 - Y) * a2 * (1.0 - a2)

dW2 = a1.T @ delta2
db2 = np.sum(delta2, axis=0)

delta1 = (delta2 @ self.W2.T) * a1 * (1.0 - a1)

dW1 = X.T @ delta1
db1 = np.sum(delta1, axis=0)
```

Les dimensions se vérifient :

| Gradient                  | Calcul                | Forme     |
| ------------------------- | --------------------- | --------- |
| $\delta_2$                | $(N,10) \odot (N,10)$ | $(N,10)$  |
| $\partial E/\partial W_2$ | $(32,N) @ (N,10)$     | $(32,10)$ |
| $\partial E/\partial b_2$ | somme sur les lignes  | $(10,)$   |
| $\delta_1$                | $(N,10) @ (10,32)$    | $(N,32)$  |
| $\partial E/\partial W_1$ | $(57,N) @ (N,32)$     | $(57,32)$ |
| $\partial E/\partial b_1$ | somme sur les lignes  | $(32,)$   |

---

## 10. Étape 6 — L'apprentissage sur le jeu de données

### Descente de gradient

À chaque mise à jour, on corrige chaque poids dans le sens qui fait baisser
l'erreur :

$$
w \leftarrow w - \eta \frac{\partial E}{\partial w}
\qquad
b \leftarrow b - \eta \frac{\partial E}{\partial b}
$$

Le signe « moins » est important :

- si la pente est négative, augmenter $w$ fait baisser $E$ ;
- donc $-\eta \times \text{pente}$ est positif, et $w$ augmente ;
- si la pente est positive, $w$ diminue.

### Mini-lots

On ne calcule pas le gradient sur les 787 morceaux d'un coup. On mélange
les données à chaque époque, puis on prend des lots de 64 morceaux :

```text
époque 1 : mélange des 787 indices
           lot 1 -> gradient -> mise à jour
           lot 2 -> gradient -> mise à jour
           ...
époque 2 : nouveau mélange
           ...
```

Avantages :

- c'est plus rapide ;
- le bruit des mini-lots aide à ne pas rester bloqué ;
- la mémoire utilisée reste petite.

### Momentum (optionnel)

La note de cadrage prévoit que le membre C compare descente simple et
momentum. Le code accepte donc un momentum $\mu$ :

$$
v \leftarrow \mu v - \eta \frac{\partial E}{\partial w}
$$

$$
w \leftarrow w + v
$$

Avec `momentum = 0`, on retrouve exactement :

$$
w \leftarrow w - \eta \frac{\partial E}{\partial w}
$$

Le momentum est donc une **extension** de la descente de gradient, pas un
changement de la rétropropagation.

### Hyperparamètres utilisés

| Hyperparamètre  |                     Valeur |
| --------------- | -------------------------: |
| Architecture    |         $57 \to 32 \to 10$ |
| Activation      |           sigmoïde partout |
| Erreur          | erreur quadratique moyenne |
| Pas $\eta$      |                        0,5 |
| Taille des lots |                         64 |
| Époques         |                        800 |
| Momentum        |        0 (descente simple) |
| Initialisation  |              Xavier/Glorot |
| Graine          |                         42 |

### Boucle d'apprentissage

```python
for epoque in range(1, epoques + 1):
    indices = rng.permutation(N)

    for debut in range(0, N, taille_lot):
        lot = indices[debut:debut + taille_lot]

        gradients, _ = self.retropropagation(X[lot], Y[lot])

        for nom, p in parametres.items():
            vitesses[nom] = momentum * vitesses[nom] - eta * gradients[nom]
            p += vitesses[nom]
```

Le coût et la précision d'apprentissage sont mesurés à chaque époque et
sauvegardés dans `resultats/historique.csv`.

---

## 11. Étape 7 — Vérification du gradient

La note de cadrage demande de vérifier le gradient par une approximation
numérique. C'est le rôle de `src/verification_gradient.py`.

### Principe

Pour un paramètre $w$, on calcule :

$$
\frac{\partial E}{\partial w}
\approx
\frac{E(w+h) - E(w-h)}{2h}
$$

C'est une **différence finie centrée**. Elle est plus précise que la
différence avant :

$$
\frac{E(w+h) - E(w)}{h}
$$

car son erreur est en $O(h^2)$ au lieu de $O(h)$.

On choisit $h = 10^{-5}$ et on compare :

- le gradient **analytique**, calculé par rétropropagation ;
- le gradient **numérique**, calculé par différences finies.

L'écart relatif utilisé est :

$$
\text{écart relatif}
=
\frac{|g_{\text{num}} - g_{\text{ana}}|}
{\max(10^{-12}, |g_{\text{num}}| + |g_{\text{ana}}|)}
$$

### Résultat obtenu

Exemple du cours (un neurone) :

```text
z                  = 0.720000
a                  = 0.672607
E                  = 0.107186
dE/da              = -0.654786
da/dz              = 0.220207
dz/dw1             = 0.800000
dE/dw1 analytique  = -0.115351
dE/dw1 numérique   = -0.115351
écart absolu       = 2.7e-11
```

Vérification sur **tous** les paramètres du réseau final
$57 \to 32 \to 10$ avec 8 exemples :

```text
Nombre de paramètres vérifiés : 2186
Coût sur le petit lot         : 0.22874071
Écart absolu maximal          : 5.187e-12
Écart relatif maximal         : 6.112e-07
Écart relatif moyen           : 3.430e-09
RÉSULTAT : gradient vérifié.
```

L'écart relatif maximal est de l'ordre de $10^{-7}$, ce qui est excellent
pour des calculs en `float64`. La rétropropagation est donc correcte.

Le rapport détaillé est aussi écrit dans
`resultats/verification_gradient.txt`.

---

## 12. Résultats de l'entraînement

Avec les hyperparamètres par défaut (graine 42) :

| Mesure                         |      Valeur |
| ------------------------------ | ----------: |
| Coût final sur l'apprentissage |    0,030815 |
| Précision apprentissage        |     84,75 % |
| Précision test (indicative)    | **72,22 %** |

Évolution du coût :

| Époque |     Coût | Précision train |
| -----: | -------: | --------------: |
|      1 | 0,127354 |          13,1 % |
|    100 | 0,066446 |          50,6 % |
|    200 | 0,056479 |          63,9 % |
|    300 | 0,049319 |          71,2 % |
|    400 | 0,043637 |          75,7 % |
|    500 | 0,039237 |          78,8 % |
|    600 | 0,035830 |          82,3 % |
|    700 | 0,033090 |          83,2 % |
|    800 | 0,030815 |          84,8 % |

Le coût baisse régulièrement : le réseau apprend bien. L'écart entre la
précision d'apprentissage (84,8 %) et la précision de test (72,2 %) indique
un début de **surapprentissage** : le réseau commence à mémoriser une partie
du jeu d'apprentissage. C'est normal avec un petit jeu de données
(787 morceaux) et 57 caractéristiques.

La précision de test de 72,2 % est **indicative** : elle est donnée pour
vérifier que le modèle fonctionne. Le calcul officiel et l'analyse de la
matrice de confusion restent la partie du membre A.

### Fichiers produits

- `data/y_pred.npy` : les 198 prédictions, dans l'ordre de `X_test` ;
- `data/y_pred.csv` : les 198 prédictions en CSV ;
- `resultats/modele.npz` : les poids et biais sauvegardés ;
- `resultats/probas_test.npy` : les 10 scores sigmoïdes pour chaque morceau
  de test ;
- `resultats/historique.csv` : coût et précision à chaque époque.

Pour recharger le modèle :

```python
from src.network import ReseauNeurones

reseau = ReseauNeurones.charger("resultats/modele.npz")
probas = reseau.predire_proba(X_test)
y_pred = reseau.predire(X_test)
```

---

## 13. Correspondance entre les notations de `CNN.md` et le code

| Notation `CNN.md`                             | Sens                | Dans le code                |
| --------------------------------------------- | ------------------- | --------------------------- |
| $x$                                           | entrées             | `X` (une ligne par morceau) |
| $w$                                           | poids               | `W1`, `W2`                  |
| $b$                                           | biais               | `b1`, `b2`                  |
| $z = w \cdot x + b$                           | somme pondérée      | `z1 = X @ W1 + b1`          |
| $a = \sigma(z)$                               | activation          | `sigmoide(z)`               |
| $y$                                           | bonne réponse       | `Y` (one-hot)               |
| $E = (a-y)^2$                                 | erreur              | `erreur(a, y)`              |
| $\partial E/\partial w$                       | pente               | `gradients["W1"]`, etc.     |
| $\eta$                                        | pas d'apprentissage | `eta`                       |
| gradient                                      | liste des pentes    | dictionnaire `gradients`    |
| $w \leftarrow w - \eta \partial E/\partial w$ | correction          | `entrainer()`               |

Le code suit donc exactement la progression des notes : neurone, activation,
erreur, pente, correction, puis rétropropagation pour calculer les pentes de
toutes les couches.

---

## 14. Limites et extensions possibles

### Ce qui est volontairement simple

- **Pas de softmax** : les 10 sorties sont des sigmoïdes indépendantes.
  C'est cohérent avec les notations de `CNN.md`. Pour une classification
  exclusive (un seul genre par morceau), softmax + entropie croisée est la
  variante la plus classique. Elle donnerait des probabilités qui somment
  à 1 et converge souvent plus vite. C'est une extension possible pour le
  membre C.
- **Pas de régularisation** (L2, dropout) : le surapprentissage est visible.
- **Pas d'arrêt précoce** : on fixe 800 époques.
- **Pas d'optimiseur avancé** : descente de gradient simple par défaut.
  Le momentum est disponible en option.
- **Pas de CNN sur spectrogrammes** : on travaille sur les 57
  caractéristiques déjà calculées, comme prévu dans la note de cadrage.

### Ce que le membre C peut tester

Le code est prévu pour être réutilisé :

```bash
# Descente simple, autre pas
.venv/bin/python src/train.py --eta 0.2 --epoques 1000

# Avec momentum
.venv/bin/python src/train.py --eta 0.1 --momentum 0.9 --epoques 600

# Autre architecture
.venv/bin/python src/train.py --cache 64 --eta 0.3 --epoques 1000
```

Les courbes peuvent être tracées à partir de `resultats/historique.csv`
(colonnes `epoque`, `cout`, `precision_train`).

### Pistes d'amélioration

1. Remplacer la sortie sigmoïde + erreur quadratique par softmax +
   entropie croisée.
2. Ajouter une régularisation L2 :

   $$
   E_{\text{tot}} = E + \frac{\lambda}{2}\sum_w w^2
   $$

3. Ajouter un arrêt précoce sur un jeu de validation.
4. Tester plusieurs architectures et plusieurs graines.
5. Faire une moyenne de plusieurs modèles (ensemble).

---

## 15. Ce qu'il faut retenir pour l'oral

1. Un neurone calcule $z = w \cdot x + b$.
2. La sigmoïde transforme $z$ en une réponse entre 0 et 1 :
   $a = \sigma(z)$.
3. L'erreur mesure l'écart : $E = (a-y)^2$.
4. La rétropropagation applique la règle de la chaîne :
   $w \to z \to a \to E$.
5. Le gradient indique comment corriger chaque poids :
   $w \leftarrow w - \eta \partial E/\partial w$.
6. La vérification numérique compare le gradient analytique à
   $(E(w+h)-E(w-h))/(2h)$ : les deux coïncident, donc la rétropropagation
   est correcte.
7. Le réseau obtient environ **72 %** de bonnes réponses sur le test, ce qui
   est un résultat honnête pour un petit réseau from scratch sur GTZAN.

---

## 16. Reproductibilité

```bash
# Vérification du gradient
.venv/bin/python src/verification_gradient.py

# Entraînement + prédictions
.venv/bin/python src/train.py

# Contrôle rapide des prédictions
.venv/bin/python - <<'PY'
import numpy as np
y_pred = np.load("data/y_pred.npy")
print(y_pred.shape, y_pred.dtype, y_pred.min(), y_pred.max())
print(np.bincount(y_pred, minlength=10))
PY
```

Tout est déterministe avec `--graine 42`. Les seules dépendances sont
Python 3 et NumPy.
