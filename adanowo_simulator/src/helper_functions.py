KG_H_TO_G_MIN = 100/6


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def calc_line_speed(stateDict: dict[str, float]) -> float:
    weight_per_area_theoretical = calc_weight_per_area_theoretical(stateDict)
    line_speed = \
        stateDict["CardMassThroughputSetpoint"] / \
        weight_per_area_theoretical / \
        stateDict["ProductWidth"] * \
        KG_H_TO_G_MIN
    return line_speed


def calc_weight_per_area_theoretical(stateDict: dict[str, float]) -> float:
    weight_per_area_theoretical = \
        stateDict["CardDeliveryWeightPerArea"] * \
        stateDict["Cross-lapperLayersCount"] * 2 / \
        prcnt_to_mult(stateDict["Needleloom2DraftRatio"]) / \
        prcnt_to_mult(stateDict["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(stateDict["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(stateDict["DrawFrameDraftRatio"])
    return weight_per_area_theoretical
