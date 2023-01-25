import numpy as np
import hydra

from src.base_classes.reward import Reward
from src.base_classes.safety_wrapper import SafetyWrapper
from src.base_classes.scenario_manager import ScenarioManager
from src.base_classes.experiment_tracker import ExperimentTracker
from src.base_classes.gym_wrapper import GymWrapper
from src.base_classes.model_wrapper import ModelWrapper
from src.base_classes.env import TrainingEnvironment
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="src", config_name="config")
def main(configuration : DictConfig):
    config=configuration.configs
    scenarioManager = ScenarioManager(config.scenarioManager.disturbanceSetting, config.scenarioManager.fibreSetting)  # changed the keys to something more clear
    print(scenarioManager.disturbanceSetting)
    scenarioManager.disturbanceSetting = {'key5': 5, 'key6': 6}
    print(scenarioManager.disturbanceSetting)
    experimentTracker = ExperimentTracker(config)
    print("Experiment Tracker", experimentTracker.metrics)
    reward = Reward(config.reward)
    print("Reward", reward._config)
    safetyWrapper = SafetyWrapper(config.safety.constraints)
    print("Safety", safetyWrapper.constraints)

    trainingEnv = TrainingEnvironment(config.env, ModelWrapper(), reward, experimentTracker, safetyWrapper)
    env = GymWrapper(env=trainingEnv)

    for _ in range(45):
        print(env.step(np.random.randn(3)))



if __name__ == "__main__":
    main()
