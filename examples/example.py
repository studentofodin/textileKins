import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import A2C

from adanowo_simulator.reward_manager import RewardManager
from adanowo_simulator.control_manager import ControlManager
from adanowo_simulator.disturbance_manager import DisturbanceManager
from adanowo_simulator.output_manager import OutputManager
from adanowo_simulator.scenario_manager import ScenarioManager
from adanowo_simulator.experiment_tracker import ExperimentTracker
from adanowo_simulator.environment import Environment
from adanowo_simulator.gym_wrapper import GymWrapper
from adanowo_simulator.reward_functions import baseline_reward

os.environ["WANDB_SILENT"] = "true"

@hydra.main(version_base=None, config_path="../config",
            config_name="main")
def main(config: DictConfig):
    reward_manager = RewardManager(baseline_reward, config.product_setup)
    disturbance_manager = DisturbanceManager(config.disturbance_setup)
    control_manager = ControlManager(config.control_setup, config.control_setup.actions_are_relative)
    output_manager = OutputManager(config.output_setup)
    scenario_manager = ScenarioManager(config.scenario_setup)
    experiment_tracker = ExperimentTracker(config.experiment_tracker, config)
    environment = Environment(config.env_setup, output_manager, reward_manager, control_manager,
                              disturbance_manager, experiment_tracker, scenario_manager)
    gym_wrapper = GymWrapper(environment, config.gym_setup)

    # check_env(gym_wrapper)
    # agent = A2C("MlpPolicy", gym_wrapper, verbose=1)
    # agent.learn(1000)

    try:
        gym_wrapper.reset()
        for _ in range(100):
            observation, _, _, _, _ = gym_wrapper.step(np.random.uniform(
                low=0.0, high=0.5, size=len(config.env_setup.used_primary_controls)))
        gym_wrapper.reset()
        for _ in range(100):
            observation, _, _, _, _ = gym_wrapper.step(np.random.uniform(
                low=0.0, high=0.5, size=len(config.env_setup.used_primary_controls)))
        gym_wrapper.close()
    except Exception as e:
        gym_wrapper.close()
        raise e

if __name__ == "__main__":
    main()
