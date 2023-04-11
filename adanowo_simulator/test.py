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
    experimentTracker = ExperimentTracker(config.experimentTracker, config)
    reward = Reward(config.product_setup)
    safetyWrapper = SafetyWrapper(OmegaConf.create({"safety": config.process_setup.safety}))
    modelWrapper = ModelWrapper(OmegaConf.merge({"pathToModels": config.env_setup.pathToModels},
                                                {"outputModels": config.env_setup.outputModels}))
    scenarioManager = ScenarioManager(config.scenario_setup)
    trainingEnv = TrainingEnvironment(OmegaConf.merge({"initialControls": config.process_setup.initialControls},
                                                      {"disturbances": config.process_setup.disturbances}), modelWrapper,
                                      reward, safetyWrapper, experimentTracker, scenarioManager, actionType=1) # actionType 0 for relative | 1 for absolute
    env = GymWrapper(trainingEnv, OmegaConf.merge({"actionSpace": config.env_setup.actionSpace},
                                                  {"observationSpace": config.env_setup.observationSpace}))

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
