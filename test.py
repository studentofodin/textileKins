import numpy as np
import hydra

from src.base_classes.reward import Reward
from src.base_classes.safety_wrapper import SafetyWrapper
from src.base_classes.scenario_manager import ScenarioManager
from src.base_classes.configuration import Configuration
from src.base_classes.experiment_tracker import ExperimentTracker
from src.base_classes.gym_wrapper import GymWrapper
from src.base_classes.model_wrapper import ModelWrapper
from src.base_classes.env import TrainingEnvironment
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="src", config_name="config")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))
    config=cfg.configs
    scenarioManager = ScenarioManager(config.scenarioManager.disturbanceSetting, config.scenarioManager.fibreSetting)  # changed the keys to something more clear
    print(scenarioManager.disturbanceSetting)
    scenarioManager.disturbanceSetting = {'key5': 5, 'key6': 6}
    print(scenarioManager.disturbanceSetting)
    experimentTracker = ExperimentTracker(config.experimentTracker.metrics)
    print("Experiment Tracker", experimentTracker.metrics)
    reward = Reward(config.reward, {'weightB' : 10.0})
    print("Reward", reward._config)
    safety = SafetyWrapper(config.safety.constraints)
    print("Safety", safety.constraints)

    trainingEnv = TrainingEnvironment(config.env, ModelWrapper(), reward, experimentTracker, np.ones(3))
    env = GymWrapper(env=trainingEnv)

    env.step(np.array([2.0,3.0,1.2]))

if __name__ == "__main__":
    main()
