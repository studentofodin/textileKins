import numpy as np
import hydra
from omegaconf import DictConfig

from src.reward import Reward
from src.safety_wrapper import SafetyWrapper
from src.experiment_tracker import ExperimentTracker
from src.gym_wrapper import GymWrapper
from src.model_wrapper import ModelWrapper
from src.env import TrainingEnvironment


@hydra.main(version_base=None, config_path="./config", config_name="main")
def main(configuration: DictConfig):

    config = configuration
    experimentTracker = ExperimentTracker(config.experimentTracker)
    reward = Reward(config.product_setup)
    safetyWrapper = SafetyWrapper(config.process_setup)
    modelWrapper = ModelWrapper(config.env_setup)
    trainingEnv = TrainingEnvironment(config, modelWrapper, reward, safetyWrapper, experimentTracker, actionType = 1) # actionType 0 for relative | 1 for absolute
    env = GymWrapper(trainingEnv, config.env_setup)

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls)))  # for actionType == absolute

    env.reset()

    for _ in range(10):
        # env.step(np.random.uniform(-0.5, 0.5, len(env.env.currentControls))) # for actionType == relative
        env.step(np.random.uniform(0, 1, len(env.env.currentControls)))  # for actionType == absolute

    env.reset()


if __name__ == "__main__":
    main()
