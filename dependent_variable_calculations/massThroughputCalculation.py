import numpy as np

G_MIN_TO_KG_H = 6 / 100


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def calculate(X: dict[str, float]) -> np.array:
    weight_per_area_theoretical = \
        X["CardDeliveryWeightPerArea"] * \
        round(X["Cross-lapperLayersCount"]) * 2 / \
        prcnt_to_mult(X["Needleloom2DraftRatio"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(X["DrawFrameDraftRatio"])

    mass_throughput = \
        X["ProductionSpeedSetpoint"] * \
        weight_per_area_theoretical * \
        X["ProductWidth"] * \
        G_MIN_TO_KG_H

    mass_throughput = np.array(mass_throughput)
    return mass_throughput
