import numpy as np
import hydra
import pathlib as pl
import yaml

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

    parentDir = pl.Path(__file__).parent

    config = configuration.configs

    scenarioManager = ScenarioManager(config.scenarioManager.disturbanceSetting, config.scenarioManager.fibreSetting)  # changed the keys to something more clear
    print(scenarioManager.disturbanceSetting)
    scenarioManager.disturbanceSetting = {'key5': 5, 'key6': 6}
    print(scenarioManager.disturbanceSetting)

    experimentTracker = ExperimentTracker(config)
    print("Experiment Tracker", experimentTracker.metrics)

    reward = Reward(config.reward)
    print("Reward", reward._config)

    safetyWrapper = SafetyWrapper(config.safety) 
    print("Safety", safetyWrapper.constraints)

    modelNames = ['unevenness_card_web', 'min_area_weight']
    modelDir = parentDir / 'models'
    modelProps = list()
    for mn in modelNames:
        with open(modelDir / (mn + '.yaml'), 'r') as stream:
            try:
                modelProps.append(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)
    modelWrapper = ModelWrapper(modelProps, modelDir)

    trainingEnv = TrainingEnvironment(config.env, modelWrapper, reward, experimentTracker, safetyWrapper)
    env = GymWrapper(env=trainingEnv)

    for _ in range(10):
        print(env.step(np.random.uniform(-1, 1, len(env.env.currentState))))




if __name__ == "__main__":
    main()
