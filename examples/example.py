import os

import numpy as np
import hydra
from omegaconf import DictConfig

from adanowo_simulator.environment_factory import EnvironmentFactory
from adanowo_simulator.gym_wrapper import GymWrapper

os.environ["WANDB_SILENT"] = "true"


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config: DictConfig):
    config.action_setup.actions_are_relative = False
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    gym_wrapper = GymWrapper(environment, config.gym_setup, config.action_setup, config.env_setup)

    gym_wrapper.reset()
    for _ in range(config.num_experiment_steps):
        observations, reward, _, _, _ = gym_wrapper.step(np.random.uniform(
            low=-2, high=2, size=len(config.env_setup.used_setpoints)))
    gym_wrapper.close()


if __name__ == "__main__":
    main()
