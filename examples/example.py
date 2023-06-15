import os
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

from src.reward_manager import RewardManager
from src.control_manager import ControlManager
from src.disturbance_manager import DisturbanceManager
from src.output_manager import OutputManager
from src.scenario_manager import ScenarioManager
from src.experiment_tracker import ExperimentTracker
from src.environment import Environment
from src.gym_wrapper import GymWrapper

os.environ['WANDB_SILENT'] = 'true'

@hydra.main(version_base=None, config_path="C:/Users/luisk/Desktop/adanowo-simulator/adanowo_simulator/config",
            config_name="main")
def main(configuration: DictConfig):

    config = configuration
    reward_manager = RewardManager(config.product_setup)
    control_manager = ControlManager(OmegaConf.merge({"actions_are_relative": config.process_setup.actions_are_relative},
                                                         {"initial_controls": config.process_setup.initial_controls},
                                                         {"control_bounds": config.process_setup.control_bounds}))
    disturbance_manager = DisturbanceManager(OmegaConf.merge({"disturbances": config.process_setup.disturbances}))
    output_manager = OutputManager(OmegaConf.merge({"path_to_models": config.env_setup.path_to_models},
                                                   {"output_models": config.env_setup.output_models},
                                                   {"outputs_are_latent": config.env_setup.outputs_are_latent},
                                                   {"observation_noise_only": config.env_setup.observation_noise_only}))
    scenario_manager = ScenarioManager(config.scenario_setup)
    experiment_tracker = ExperimentTracker(config.experiment_tracker, config)
    environment = Environment(OmegaConf.create(), output_manager, reward_manager, control_manager,
                              disturbance_manager, experiment_tracker, scenario_manager)
    gym_wrapper = GymWrapper(environment, OmegaConf.merge({"action_space": config.env_setup.action_space},
                                                   {"observation_space": config.env_setup.observation_space}))

    # check_env(gym_wrapper)
    # agent = A2C("MlpPolicy", gym_wrapper, verbose=1)
    # agent.learn(1000)

    for _ in range(1):
        observation, _, _, _, _ = gym_wrapper.step(np.random.uniform(low=0.0, high=0.5, size=gym_wrapper.environment.control_manager.n_controls))
    observation, _ = gym_wrapper.reset()

    pass


if __name__ == "__main__":
    main()
