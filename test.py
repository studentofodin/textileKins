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


def main():
    scenarioManager = ScenarioManager({'dist1': 1, 'dist2': 2},
                                      {'fibre3': 3, 'fibre4': 4})  # changed the keys to something more clear
    scenarioManager.disturbanceSetting = {'key5': 5, 'key6': 6}
    scenarioManager.fibreSetting = {'key7': 7, 'key8': 8}
    configuration = Configuration(requirements={'req1': 123, 'req2': 321},
                                  actorConstraints={'constraint1': 4, 'constraint2': 5},
                                  productionScenario={'pS1': 2, 'pS2': 5},
                                  actionParams={'ap1': 4, 'ap2': 5},
                                  stateParams={'sP1': 3, 'sP2': 4},
                                  stepsUntilLabDataAvailable=100,
                                  observationParams={'oP1': 4, 'oP2': 9},
                                  maxSteps=500)
    configuration.requirements = {'New Req1': 222, 'New Req2': 111}
    experimentTracker = ExperimentTracker(['Reward', 'Weight Per Area', 'Metric3'])
    reward = Reward({'req': 5})
    safety = SafetyWrapper({'constraint': 2})

    trainingEnv = TrainingEnvironment(configuration, ModelWrapper(), reward, experimentTracker, np.ones(3))
    env = GymWrapper(env=trainingEnv)

@hydra.main(version_base=None, config_path="src", config_name="config")
def config(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
    config()
