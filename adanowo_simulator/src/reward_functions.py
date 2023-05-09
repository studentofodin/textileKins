KG_H_TO_G_MIN = 100/6
MIN_TO_H = 60


def prcnt_to_mult(prcnt: float) -> float:
    return (prcnt / 100) + 1


def baseline_reward(state: dict[str, float], outputs: dict[str, float], config) -> float:
    weight_per_area_theoretical = \
        state["CardDeliveryWeightPerArea"] * \
        state["Cross-lapperLayersCount"] * 2 / \
        prcnt_to_mult(state["Needleloom1DraftRatioIntake"]) / \
        prcnt_to_mult(state["Needleloom1DraftRatio"]) / \
        prcnt_to_mult(state["DrawFrameDraftRatio"])
    line_speed = \
        state["CardMassThroughputSetpoint"] / \
        weight_per_area_theoretical / \
        state["ProductWidth"] * \
        KG_H_TO_G_MIN

    # material costs
    material_costs = state["CardMassThroughputSetpoint"] * config.fibre_costs
    # energy costs
    energy_costs = outputs["linePowerConsumption"] * config.energy_costs
    # Production income
    income = config.selling_price * line_speed * MIN_TO_H * state["ProductWidth"]

    # Component 1: Calculate economic efficiency
    contribution_margin = income - energy_costs - material_costs

    # Component 2: card floor evenness
    floor_quality = outputs["cardWebUnevenness"] * config.floor_quality_weight

    reward = contribution_margin - floor_quality

    return reward
