from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.environment_factory import EnvironmentFactory

UNIT_STEP = 1
CONFIG_DIR_RELATIVE = "test_config"
CONFIG_NAME = "main"


def get_test_env():
    with initialize(version_base=None, config_path="test_config"):
        config = compose(config_name=CONFIG_NAME)
        config.action_setup.actions_are_relative = True
        factory = EnvironmentFactory(config)
        environment = factory.create_environment()
        environment.reset()
        return environment, config


test_env, test_config = get_test_env()


# Test set 1: Test correctness of model inputs

# Reference values
reference_setpoints = (OmegaConf.to_container(test_config.action_setup.initial_setpoints, resolve=True))
reference_disturbances = (OmegaConf.to_container(test_config.disturbance_setup.disturbances, resolve=True))
reference_state_without_dependent = reference_setpoints | reference_disturbances


def test_state_correctly_compiled():
    pass


def test_step_correctly_calculated():
    pass


def test_setpoint_violation_correctly_prevented():
    pass


def test_dependent_violation_correctly_prevented():
    pass


def test_scenario_random():
    pass


#test_state = reference_setpoints | reference_disturbances  #  | quality_bounds

# test cases for model input values

# Test set 2: Test correctness of model transformations


def test_model_unevenness_transformation():
    pass


# Test set 3: Test correctness of model outputs

def test_model_unevenness_output():
    pass

# Test set 4: Test correctness of env setup


def test_reset_successful():
    pass


def test_parall_elexecution_results():
    pass


test_env.close()
