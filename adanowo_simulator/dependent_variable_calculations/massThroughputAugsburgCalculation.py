import numpy as np

G_MIN_TO_KG_H = 6 / 100
CARD_WIDTH = 2.5  # m


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def calculate(X: dict) -> np.array:
    X = X.copy()
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)

    mass_throughput = \
        X["CardDeliveryWeightPerArea"] * \
        X["CARD_WIDTH"] * \
        X["CardDeliverySpeed"] * \
        G_MIN_TO_KG_H

    mass_throughput = np.array(mass_throughput).flatten()
    return mass_throughput
