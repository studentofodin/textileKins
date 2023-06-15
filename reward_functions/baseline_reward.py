from omegaconf import DictConfig

H_TO_MIN = 60


def get_reward(controls: dict[str, float], disturbances: dict[str, float], outputs: dict[str, float],
                    config: DictConfig) -> float:

    # material costs
    material_costs = controls["CardMassThroughputSetpoint"] * config.fibre_costs
    # energy costs
    energy_costs = outputs["linePowerConsumption"] * config.energy_costs
    # Production income
    income = config.selling_price * outputs["lineSpeed"] * H_TO_MIN * disturbances["ProductWidth"]

    # Component 1: Calculate economic efficiency
    contribution_margin = income - energy_costs - material_costs

    # Component 2: card floor evenness
    floor_quality = outputs["cardWebUnevenness"] * config.floor_quality_weight

    reward = contribution_margin - floor_quality

    return reward
