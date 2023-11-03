import pytest

from hydra import initialize, compose
from omegaconf import OmegaConf

from adanowo_simulator.environment_factory import EnvironmentFactory

UNIT_STEP = 1
CONFIG_DIR_RELATIVE = "test_config"
CONFIG_NAME = "main"


@pytest.fixture(scope="function")
def config():
    with initialize(version_base=None, config_path="test_config"):
        config = compose(config_name=CONFIG_NAME)
        config.action_setup.actions_are_relative = False
        return config


@pytest.fixture(scope="function")
def test_env(config):
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    environment.reset()
    yield environment
    environment.close()
    return environment


@pytest.fixture(scope="function")
def reference_values(config):
    reference_setpoints = OmegaConf.to_container(config.action_setup.initial_setpoints, resolve=True)
    reference_disturbances = OmegaConf.to_container(config.disturbance_setup.disturbances, resolve=True)
    reference_state_without_dependent = reference_setpoints | reference_disturbances
    return_dict = {
        "reference_setpoints": reference_setpoints,
        "reference_disturbances": reference_disturbances,
        "reference_state_without_dependent": reference_state_without_dependent
    }
    return return_dict


@pytest.fixture(scope="function")
def step_values(reference_values):
    reference_setpoints = reference_values["reference_setpoints"]
    zero_step = reference_setpoints
    unit_step = {key: value + UNIT_STEP for key, value in reference_setpoints.items()}
    return_dict = {
        "zero_step": zero_step,
        "unit_step": unit_step
    }
    return return_dict


# Test set 1: Test correctness of model inputs
def test_state_data_types(test_env, step_values):
    zero_step = step_values["zero_step"]

    _, state, _, _ = test_env.step(zero_step)
    for key, value in state.items():
        assert isinstance(key, str), f"Key '{key}' is not a string."
        assert isinstance(value, float) or isinstance(value, int), f"value '{key}' is not int or float"


def test_state_keys(test_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    _, state, _, _ = test_env.step(zero_step)

    for key in reference_state_without_dependent.keys():
        assert key in state.keys(), f"Key '{key}' from reference not found in the state."


def test_state_values(test_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = test_env.step(zero_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' initiated with wrong value."
    assert reward > 0


def test_step_parallel_processing(test_env, reference_values, step_values, config):
    config.parallel_execution = True
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    environment.reset()

    unit_step = step_values["unit_step"]
    reference_setpoints = reference_values["reference_setpoints"]
    reward, state, _, _ = test_env.step(unit_step)

    for key, value in reference_setpoints.items():
        assert pytest.approx(value + UNIT_STEP) == state[key], (f"Key '{key}' has wrong value after unit step "
                                                                f"(Parallel execution).")
    environment.close()


def test_step(test_env, reference_values, step_values):
    unit_step = step_values["unit_step"]
    reference_setpoints = reference_values["reference_setpoints"]
    reward, state, _, _ = test_env.step(unit_step)

    for key, value in reference_setpoints.items():
        assert pytest.approx(value + UNIT_STEP) == state[key], f"Key '{key}' has wrong value after unit step."


def test_setpoint_violation_prevented(test_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    huge_step = {key: value + 1000 for key, value in zero_step.items()}
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = test_env.step(huge_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' has has changed after setpoint constraint violation."
    assert reward < 0


def test_dependent_violation_prevented(test_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    zero_step["Cross-lapperLayersCount"] = 17
    zero_step["CardDeliveryWeightPerArea"] = 77
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = test_env.step(zero_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' has has changed after dependent constraint violation."
    assert reward < 0


def test_scenarios(test_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    reference_disturbances = reference_values["reference_disturbances"]
    _, state_2, _, quality_bounds_2 = test_env.step(zero_step)
    assert test_env._step_index == 2
    for _ in range(3):  # index should be 5
        _, _, _, quality_bounds_5 = test_env.step(zero_step)
    assert test_env._step_index == 5
    for _ in range(5):  # index should be 10
        _, state_10, _, quality_bounds_10 = test_env.step(zero_step)
    assert test_env._step_index == 10

    # test random output bound setting that changes every 5 steps
    assert not (pytest.approx(quality_bounds_2["AreaWeightLane1"]["upper"]) ==
                quality_bounds_5["AreaWeightLane1"]["upper"]), "No random output bound change after 5 time steps"
    assert not (pytest.approx(quality_bounds_2["AreaWeightLane1"]["upper"]) ==
                quality_bounds_10["AreaWeightLane1"]["upper"]), "No random output bound change after 10 time steps"

    # test deterministic output bound change after 10 steps
    assert pytest.approx(quality_bounds_2["AreaWeightLane2"]["lower"]) == 100.0
    assert pytest.approx(quality_bounds_10["AreaWeightLane2"]["lower"]) == 350.0, ("No deterministic output "
                                                                                   "bound change after 10 time steps")

    # test deterministic disturbance change after 10 steps
    assert pytest.approx(state_2["CalenderTemperature"]) == 174.0
    assert pytest.approx(state_10["CalenderTemperature"]) == 160.0, ("No deterministic disturbance change after "
                                                                     "10 time steps")

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
