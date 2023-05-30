MIN_TO_H = 60


def baseline_reward(state: dict[str, float], outputs: dict[str, float], config) -> float:

    # material costs
    material_costs = state["CardMassThroughputSetpoint"] * config.fibre_costs
    # energy costs
    energy_costs = outputs["linePowerConsumption"] * config.energy_costs
    # Production income
    income = config.selling_price * outputs["lineSpeed"] * MIN_TO_H * state["ProductWidth"]

    # Component 1: Calculate economic efficiency
    contribution_margin = income - energy_costs - material_costs

    # Component 2: card floor evenness
    floor_quality = outputs["cardWebUnevenness"] * config.floor_quality_weight

    reward = contribution_margin - floor_quality

    return reward.item()
