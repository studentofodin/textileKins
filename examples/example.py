import os

import numpy as np
import hydra
from omegaconf import DictConfig

from adanowo_simulator.environment_factory import EnvironmentFactory
from adanowo_simulator.gym_wrapper import GymWrapper

os.environ["WANDB_SILENT"] = "true"


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config: DictConfig):
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    gym_wrapper = GymWrapper(environment, config.gym_setup, config.env_setup)

    gym_wrapper.reset()
    for _ in range(config.num_experiment_steps):
        _, _, _, _, _ = gym_wrapper.step(np.random.uniform(
            low=-0.25, high=0.5, size=len(config.env_setup.used_setpoints)))
    gym_wrapper.reset()
    gym_wrapper.close()


if __name__ == "__main__":
    main()
