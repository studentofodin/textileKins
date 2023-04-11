import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from src.reward import Reward
from src.state_manager import StateManager
from src.experiment_tracker import ExperimentTracker
from src.gym_wrapper import GymWrapper
from src.model_wrapper import ModelWrapper
from src.scenario_manager import ScenarioManager
from src.env import TrainingEnvironment

@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(configuration: DictConfig):

    config = configuration
    experimentTracker = ExperimentTracker(config.experimentTracker, config)
    reward = Reward(config.product_setup)
    stateManager = StateManager(config.process_setup, actionType=1) # actionType 0 for relative | 1 for absolute
    modelWrapper = ModelWrapper(OmegaConf.merge({"pathToModels": config.env_setup.pathToModels},
                                                {"outputModels": config.env_setup.outputModels}))
    scenarioManager = ScenarioManager(config.scenario_setup)
    trainingEnv = TrainingEnvironment({}, modelWrapper, reward, stateManager, experimentTracker, scenarioManager)
    env = GymWrapper(trainingEnv, OmegaConf.merge({"actionSpace": config.env_setup.actionSpace},
                                                  {"observationSpace": config.env_setup.observationSpace}))

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, env.env.stateManager.n_controls)) # for actionType == relative
        env.step(np.random.uniform(0, 1, env.env.stateManager.n_controls))  # for actionType == absolute

    env.reset()
    pass

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, env.env.stateManager.n_controls)) # for actionType == relative
        env.step(np.random.uniform(0, 1, env.env.stateManager.n_controls))  # for actionType == absolute

    pass


if __name__ == "__main__":
    main()
