import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


from src.reward_manager import RewardManager
from src.control_manager import ControlManager
from src.disturbance_manager import DisturbanceManager
from src.model_wrapper import ModelWrapper
from src.scenario_manager import ScenarioManager
from src.experiment_tracker import ExperimentTracker
from src.environment import TrainingEnvironment
from src.gym_wrapper import GymWrapper
from src.reward_functions import baseline_reward


@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(configuration: DictConfig):

    config = configuration
    reward_manager = RewardManager(config.product_setup, baseline_reward)
    control_manager = ControlManager(OmegaConf.merge({"actions_are_relative": config.process_setup.actions_are_relative},
                                                         {"initial_controls": config.process_setup.initial_controls},
                                                         {"control_bounds": config.process_setup.control_bounds}))
    disturbance_manager = DisturbanceManager(OmegaConf.merge({"disturbances": config.process_setup.disturbances}))
    model_wrapper = ModelWrapper(OmegaConf.merge({"path_to_models": config.env_setup.path_to_models},
                                                     {"output_models": config.env_setup.output_models}))
    scenario_manager = ScenarioManager(config.scenario_setup)
    experiment_tracker = ExperimentTracker(config.experiment_tracker, config)
    training_env = TrainingEnvironment(OmegaConf.create(), model_wrapper, reward_manager, control_manager,
                                       disturbance_manager, experiment_tracker, scenario_manager)
    env = GymWrapper(training_env, OmegaConf.merge({"action_space": config.env_setup.action_space},
                                                   {"observation_space": config.env_setup.observation_space}))

    for _ in range(100):
        env.step(np.random.uniform(low=0.0, high=0.5, size=env.env.control_manager.n_controls))
    env.reset()
    pass


if __name__ == "__main__":
    main()
