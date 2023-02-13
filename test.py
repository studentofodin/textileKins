import numpy as np
import hydra
from omegaconf import DictConfig

from src.base_classes.reward import Reward
from src.base_classes.safety_wrapper import SafetyWrapper
from src.base_classes.experiment_tracker import ExperimentTracker
from src.base_classes.gym_wrapper import GymWrapper
from src.base_classes.model_wrapper import ModelWrapper
from src.base_classes.env import TrainingEnvironment


@hydra.main(version_base=None, config_path="src", config_name="config")
def main(configuration: DictConfig):

    config = configuration.configs
    experimentTracker = ExperimentTracker(config.experimentTracker)
    reward = Reward(config.reward)
    safetyWrapper = SafetyWrapper(config.safetyWrapper)
    modelWrapper = ModelWrapper(config.modelWrapper)
    trainingEnv = TrainingEnvironment(config.env, modelWrapper, reward, safetyWrapper, experimentTracker)
    env = GymWrapper(trainingEnv)

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls)))  # for actionType == absolute

    env.reset()

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls))) # for actionType == absolute



if __name__ == "__main__":
    main()
