from omegaconf import DictConfig
from sklearn.base import BaseEstimator, TransformerMixin

H_TO_MIN = 60


class ParameterizedStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def fit(self, X: float, y=None):
        # In this custom implementation, fit does nothing but return itself.
        return self

    def transform(self, X: float, y=None) -> float:
        # Apply the transformation: (X - mean) / std
        return (X - self.mean) / self.std

    def inverse_transform(self, X: float, y=None) -> float:
        # Reverse the transformation: X * std + mean
        return X * self.std + self.mean


def baseline_objective(state: dict[str, float], outputs: dict[str, float], config: DictConfig) -> float:
    # material costs
    material_costs = state["MassThroughput"] * config.fibre_costs
    # energy costs
    energy_costs = outputs["LinePowerConsumption"] * config.energy_costs
    # Production income
    income = config.selling_price * state["ProductionSpeed"] * H_TO_MIN * state["ProductWidth"]

    # Component 1: Calculate economic efficiency
    contribution_margin = income - energy_costs - material_costs

    # Component 2: card floor evenness
    scaler = ParameterizedStandardScaler(config.signal_mean, config.signal_std)
    scaled_signal = scaler.transform(outputs["NonwovenUnevenness"])
    floor_quality = scaled_signal * config.floor_quality_weight

    reward = contribution_margin - floor_quality

    return reward


def baseline_penalty(state: dict[str, float], outputs: dict[str, float], config: DictConfig) -> float:
    # material costs
    material_costs = state["MassThroughput"] * config.fibre_costs
    # energy costs
    energy_costs = outputs["LinePowerConsumption"] * config.energy_costs

    # Calculate economic loss
    penalty = - energy_costs - material_costs

    return penalty
