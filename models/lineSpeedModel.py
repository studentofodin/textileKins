import numpy as np

KG_H_TO_G_MIN = 100 / 6


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def model(X: dict) -> [np.array, np.array]:
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)

    weight_per_area_theoretical = \
        X["CardDeliveryWeightPerArea"] * \
        X["Cross-lapperLayersCount"] * 2 / \
        prcnt_to_mult(X["Needleloom2DraftRatio"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(X["DrawFrameDraftRatio"])

    line_speed = \
        X["CardMassThroughputSetpoint"] / \
        weight_per_area_theoretical / \
        X["ProductWidth"] * \
        KG_H_TO_G_MIN

    return line_speed, np.zeros(1)
