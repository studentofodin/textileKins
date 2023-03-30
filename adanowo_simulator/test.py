import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.reward import Reward
from src.safety_wrapper import SafetyWrapper
from src.experiment_tracker import ExperimentTracker
from src.gym_wrapper import GymWrapper
from src.model_wrapper import ModelWrapper
from src.scenario_manager import ScenarioManager
from src.env import TrainingEnvironment



@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(configuration: DictConfig):

    config = configuration
    experimentTracker = ExperimentTracker(config.experimentTracker)
    reward = Reward(config.product_setup)
    safetyWrapper = SafetyWrapper(config.process_setup)
    modelWrapper = ModelWrapper(config.env_setup, config.env_setup.usedOutputs)
    scenarioManager = ScenarioManager(config.scenario_setup, modelWrapper._config)
    trainingEnv = TrainingEnvironment(config, modelWrapper, reward, safetyWrapper, experimentTracker, scenarioManager,
                                      actionType=1) # actionType 0 for relative | 1 for absolute
    env = GymWrapper(trainingEnv, config.env_setup)

    # configcopy = config.copy()
    # productsetup = config.product_setup
    # config.product_setup.penalty = 0
    # print(config.product_setup.penalty,"and",productsetup.penalty)
    # productsetup.penalty = 1
    # print(config.product_setup.penalty, "and", productsetup.penalty)
    # config.update(configcopy)
    # for key in config.keys():
    #     config[key] = configcopy[key]
    # for key in config.product_setup.keys():
    #     config.product_setup[key] = configcopy.product_setup[key]
    # print(config.product_setup.penalty,"and",productsetup.penalty)

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls)))  # for actionType == absolute

    env.reset()
    pass

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls)))  # for actionType == absolute

    pass


if __name__ == "__main__":
    main()
