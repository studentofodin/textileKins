import numpy as np

from adanowo_simulator.abstract_base_class.secondary_control_model import  AbstractSecondaryControlModel

KG_H_TO_G_MIN = 100/6

def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1

class SecondaryControlModel(AbstractSecondaryControlModel):

    def calculate_control(X: dict[str, float]) -> np.array:

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

        return np.array(line_speed)
