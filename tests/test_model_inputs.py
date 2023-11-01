import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from adanowo_simulator.environment_factory import EnvironmentFactory

UNIT_STEP = 1


@hydra.main(version_base=None, config_path="./test_config", config_name="main")
def get_test_env(config: DictConfig):
    config.action_setup.actions_are_relative = True
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    environment.reset()
    return environment


test_env = get_test_env()

### Test set 1: Test correctness of model inputs

# Reference values
reference_setpoints: DictConfig = OmegaConf.load(Path("./test_config/action_setup/test.yaml"))
reference_disturbances: DictConfig = OmegaConf.load(Path("./test_config/disturbance_setup/test.yaml"))


def test_state_correctly_compiled():
    pass


def test_step_correctly_calculated():
    pass


def test_setpoint_violation_correctly_prevented():
    pass


def test_dependent_violation_correctly_prevented():
    pass


def test_scenario_random():
    #test all kinds of scenarios
    pass

test_state = reference_setpoints | reference_disturbances  #  | quality_bounds

# test cases for model input values

### Test set 2: Test correctness of model transformations


### Test set 3: Test correctness of model outputs


test_env.close()