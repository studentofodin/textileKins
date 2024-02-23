from pathlib import Path

from hydra import initialize, compose
import cma  # requires https://github.com/CMA-ES/pycma
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.environment_factory import EnvironmentFactory
from adanowo_simulator.gym_wrapper import GymWrapper

MAX_ITERATIONS = 100
initial_std_dev = 0.06  # Adjust based on the variability of actions


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(config: DictConfig):
    # initialise environment
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    environment_wrapped = GymWrapper(environment, config.gym_setup, config.action_setup, config.env_setup)

    # define objective function
    def objective_function(actions: np.array) -> float:
        _, reward, _, _, _ = environment_wrapped.step(actions)
        return - float(reward)

    # reset environment
    initial_setpoints, _ = environment_wrapped.reset()

    # init cma-es
    initial_mean = initial_setpoints

    es = cma.CMAEvolutionStrategy(initial_mean, initial_std_dev)

    # Loop over a defined number of iterations or until convergence
    iteration = 0
    while True:
        if iteration >= MAX_ITERATIONS or es.stop():
            print("Optimisation finished.")
            break

        solutions = es.ask()  # Get a batch of candidate solutions (actions)
        rewards = [objective_function(action) for action in solutions]  # Evaluate each solution
        es.tell(solutions, rewards)  # Update the optimizer with results
        es.disp()

        iteration += 1
    # Make sure the environment gets closed properly!
    environment_wrapped.close()


if __name__ == "__main__":
    main()
