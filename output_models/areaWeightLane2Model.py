import numpy as np

VARIANCE_AT_100_GSM = 49  # (g per sqm)^2
GSM_100 = 100  # g per sqm


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def model(X: dict) -> [np.array, np.array]:
    X = X.copy()
    for key in X.keys():
        X[key] = np.array(X[key]).reshape(-1, 1)

    weight_per_area_theoretical = \
        X["CardDeliveryWeightPerArea"] * \
        X["Cross-lapperLayersCount"].round() / \
        prcnt_to_mult(X["Needleloom2DraftRatio"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(X["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(X["DrawFrameDraftRatio"])

    var = VARIANCE_AT_100_GSM * np.power(weight_per_area_theoretical / GSM_100, 2)

    return weight_per_area_theoretical, var
