import os
from copy import copy

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.environment_factory import EnvironmentFactory

os.environ["WANDB_SILENT"] = "true"


@hydra.main(version_base=None, config_path="../config", config_name="paper")
def main(config: DictConfig):
    config.action_setup.actions_are_relative = False
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()

    initial_setpoints = OmegaConf.to_container(config.action_setup.initial_setpoints, resolve=True)

    # First episode
    environment.reset()
    action = copy(initial_setpoints)
    production_speeds = np.linspace(8, 17, config.num_experiment_steps).flatten()
    card_web = np.linspace(76, 42, config.num_experiment_steps*2).flatten()
    layers = np.linspace(6, 3, config.num_experiment_steps*2).flatten()
    for i in range(config.num_experiment_steps):
        action["ProductionSpeedSetpoint"] = float(production_speeds[i])
        _, _, _, _ = environment.step(action)

    action = copy(initial_setpoints)

    for i in range(config.num_experiment_steps*2):
        action["CardDeliveryWeightPerArea"] = float(card_web[i])
        action["Cross-lapperLayersCount"] = float(layers[i])
        _, _, _, _ = environment.step(action)

    environment.close()


if __name__ == "__main__":
    main()
