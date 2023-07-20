import numpy as np

KG_H_TO_G_MIN = 100 / 6


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def calculate(X: dict[str, float]) -> np.array:
    weight_per_area_theoretical = \
        X["CardDeliveryWeightPerArea"] * \
        X["Cross-lapperLayersCount"].round() * 2 / \
        prcnt_to_mult(X["Needleloom2DraftRatio"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(X["DrawFrameDraftRatio"])

    line_speed = \
        X["CardMassThroughputSetpoint"] / \
        weight_per_area_theoretical / \
        X["ProductWidth"] * \
        KG_H_TO_G_MIN

    line_speed = np.array(line_speed)
    return line_speed
