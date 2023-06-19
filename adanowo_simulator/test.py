import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.reward_manager import RewardManager
from src.state_manager import StateManager
from src.experiment_tracker import ExperimentTracker
from src.gym_wrapper import GymWrapper
from src.model_manager import ModelManager
from src.scenario_manager import ScenarioManager
from src.environment import TrainingEnvironment
from src.reward_functions import baseline_reward


@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(configuration: DictConfig):
    config = configuration
    experiment_tracker = ExperimentTracker(config.experiment_tracker, config)
    reward_manager = RewardManager(config.product_setup, baseline_reward)
    state_manager = StateManager(config.process_setup, relative_actions=True)
    model_wrapper = ModelManager(OmegaConf.merge({"path_to_models": config.env_setup.path_to_models},
                                                 {"output_models": config.env_setup.output_models}))
    scenario_manager = ScenarioManager(config.scenario_setup)
    training_env = TrainingEnvironment(OmegaConf.create(), model_wrapper, reward_manager, state_manager,
                                       experiment_tracker, scenario_manager)
    env = GymWrapper(training_env, OmegaConf.merge({"action_space": config.env_setup.action_space},
                                                   {"observation_space": config.env_setup.observation_space}))

    #for _ in range(100):
    #    env.step(np.random.uniform(low=-0.1, high=0.4, size=env.env.state_manager.n_controls))
    #env.reset()

    for _ in range(1000):
        env.step(np.random.uniform(low=-0.1, high=0.4, size=env.env.state_manager.n_controls))
    env.shutdown()


if __name__ == "__main__":
    main()
