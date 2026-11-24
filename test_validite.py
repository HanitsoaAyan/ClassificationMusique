import numpy as np

from src.network import ReseauNeurones
from src.optimisation import entrainer

X = np.load("data/X_train.npy")
y = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

reseau = ReseauNeurones(57, 32, 10, graine=42)
couts, precisions = entrainer(reseau, X, y, eta=0.5, epoques=800, momentum=0.0)

print("coût final       :", round(couts[-1], 6), "(B : 0.030815)")
print("précision train  :", round(precisions[-1], 4), "(B : 0.8475)")
print("précision test   :", round(float(np.mean(reseau.predire(X_test) == y_test)), 4), "(B : 0.7222)")
