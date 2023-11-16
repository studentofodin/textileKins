import pytest

from hydra import initialize, compose
from omegaconf import OmegaConf

from adanowo_simulator.environment_factory import EnvironmentFactory
from adanowo_simulator.gym_wrapper import GymWrapper
import adanowo_simulator.transformations as transformations

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
def get_env(config):
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
def test_state_data_types(get_env, step_values):
    zero_step = step_values["zero_step"]

    _, state, _, _ = get_env.step(zero_step)
    for key, value in state.items():
        assert isinstance(key, str), f"Key '{key}' is not a string."
        assert isinstance(value, float) or isinstance(value, int), f"value '{key}' is not int or float"


def test_state_keys(get_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    _, state, _, _ = get_env.step(zero_step)

    for key in reference_state_without_dependent.keys():
        assert key in state.keys(), f"Key '{key}' from reference not found in the state."


def test_state_values(get_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = get_env.step(zero_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' initiated with wrong value."


def test_step(get_env, reference_values, step_values):
    unit_step = step_values["unit_step"]
    reference_setpoints = reference_values["reference_setpoints"]
    reward, state, _, _ = get_env.step(unit_step)
    for key, value in reference_setpoints.items():
        assert pytest.approx(value + UNIT_STEP) == state[key], f"Key '{key}' has wrong value after unit step."


def test_setpoint_violation_prevented(get_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    huge_step = {key: value + 1000 for key, value in zero_step.items()}
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = get_env.step(huge_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' has has changed after setpoint constraint violation."
    assert reward < 0


def test_dependent_violation_prevented(get_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    zero_step["Cross-lapperLayersCount"] = 8
    zero_step["CardDeliveryWeightPerArea"] = 77
    reference_state_without_dependent = reference_values["reference_state_without_dependent"]
    reward, state, _, _ = get_env.step(zero_step)

    for key, value in reference_state_without_dependent.items():
        assert pytest.approx(value) == state[key], f"Key '{key}' has has changed after dependent constraint violation."
    assert reward < 0


def test_scenarios(get_env, reference_values, step_values, config):
    zero_step = step_values["zero_step"]
    intial_calender_temperature = reference_values["reference_disturbances"]["CalenderTemperature"]
    initial_area_weight_2_lower_bound = config.objective_setup.output_bounds["AreaWeightLane2"]["lower"]
    area_weight_2_lower_10 = config.scenario_setup.output_bounds["AreaWeightLane2"]["lower"][0][1]
    calender_temperature_10 = config.scenario_setup.disturbances["CalenderTemperature"][0][1]

    _, state_2, _, quality_bounds_2 = get_env.step(zero_step)
    assert get_env._step_index == 2
    for _ in range(3):  # index should be 5
        _, _, _, quality_bounds_5 = get_env.step(zero_step)
    assert get_env._step_index == 5
    for _ in range(5):  # index should be 10
        _, state_10, _, quality_bounds_10 = get_env.step(zero_step)
    assert get_env._step_index == 10

    # test random output bound setting that changes every 5 steps
    assert not (pytest.approx(quality_bounds_2["AreaWeightLane1"]["upper"]) ==
                quality_bounds_5["AreaWeightLane1"]["upper"]), "No random output bound change after 5 time steps"
    assert not (pytest.approx(quality_bounds_2["AreaWeightLane1"]["upper"]) ==
                quality_bounds_10["AreaWeightLane1"]["upper"]), "No random output bound change after 10 time steps"

    # test deterministic output bound change after 10 steps
    assert pytest.approx(quality_bounds_2["AreaWeightLane2"]["lower"]) == initial_area_weight_2_lower_bound
    assert pytest.approx(quality_bounds_10["AreaWeightLane2"]["lower"]) == area_weight_2_lower_10, \
        "No deterministic outputbound change after 10 time steps"

    # test deterministic disturbance change after 10 steps
    assert pytest.approx(state_2["CalenderTemperature"]) == intial_calender_temperature
    assert pytest.approx(state_10["CalenderTemperature"]) == calender_temperature_10, \
        "No deterministic disturbance change after 10 time steps"


# Test set 2: Test correctness of model transformations

def test_model_unevenness_transformation(get_env, step_values, reference_values):
    training_features = ["FG_soll", "mean_mass_cylinders", "Diff_ArbeiterZuWender"]
    zero_step = step_values["zero_step"]
    _, test_state, _, _ = get_env.step(zero_step)
    unpack_dict = get_env.output_manager._output_models["CardWebUnevenness"]._unpack_func
    transformed_state = unpack_dict(test_state, training_features)

    assert pytest.approx(transformed_state[0][0]) == test_state["CardDeliveryWeightPerArea"] * 0.160, \
        "FG_soll transformation is wrong."

    assert pytest.approx(transformed_state[0][1], abs=0.001) == 0.2938, "mean_mass_cylinders transformation is wrong."

    assert pytest.approx(transformed_state[0][2]) == test_state["v_WorkerMain"] - test_state["v_StripperMain"], \
        "Diff_ArbeiterZuWender transformation is wrong."


# Test set 3: Test correctness of model outputs

def test_model_unevenness_output(get_env, reference_values):
    reference_values = reference_values["reference_state_without_dependent"]
    model = get_env.output_manager._output_models["CardWebUnevenness"]
    reference_values["MassThroughput"] = 900

    # test low prediction
    reference_values["CardDeliveryWeightPerArea"] = 41.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert mean_pred.flatten()[0] <= 0.0, "Mean prediction too high."
    # test high prediction
    reference_values["CardDeliveryWeightPerArea"] = 77.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert mean_pred.flatten()[0] >= 0.0, "Mean prediction too low."


def test_model_power_consumption(get_env, reference_values):
    reference_values = reference_values["reference_state_without_dependent"]
    model = get_env.output_manager._output_models["LinePowerConsumption"]

    # test low prediction
    reference_values["MassThroughput"] = 600
    reference_values["Needleloom1FeedPerStroke"] = 11.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 315.0, "Low prediction is wrong."
    # test high prediction
    reference_values["MassThroughput"] = 1200
    reference_values["Needleloom1FeedPerStroke"] = 10.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 345.0, "High prediction is wrong."


def test_model_tensile_strength_CD(get_env, reference_values):
    reference_values = reference_values["reference_state_without_dependent"]
    model = get_env.output_manager._output_models["TensileStrengthCD"]

    # test low prediction
    reference_values["Cross-lapperLayersCount"] = 2.0
    reference_values["Needleloom1FeedPerStroke"] = 12.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 182.0, "Low prediction is wrong."
    # test high prediction
    reference_values["Cross-lapperLayersCount"] = 8.0
    reference_values["Needleloom1FeedPerStroke"] = 10.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 955.0, "High prediction is wrong."
    # test different fibre mixture
    reference_values["FibreA"] = 1.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 1226.0, "FibreA prediction is wrong."


def test_model_tensile_strength_MD(get_env, reference_values):
    reference_values = reference_values["reference_state_without_dependent"]
    model = get_env.output_manager._output_models["TensileStrengthMD"]

    # test low prediction
    reference_values["Cross-lapperLayersCount"] = 2.0
    reference_values["Needleloom1FeedPerStroke"] = 12.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 154.0, "Low prediction is wrong."
    # test high prediction
    reference_values["Cross-lapperLayersCount"] = 8.0
    reference_values["Needleloom1FeedPerStroke"] = 10.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 710.0, "High prediction is wrong."
    # test different fibre mixture
    reference_values["FibreA"] = 1.0
    mean_pred, _ = model.predict_y(reference_values, observation_noise_only=True)
    assert pytest.approx(mean_pred.flatten()[0], abs=5) == 975.0, "High prediction is wrong."


# Test set 4: Test correctness of reward calculation
def test_reward_without_violations(get_env, reference_values, step_values):
    unit_step = step_values["unit_step"]
    reward, state, _, _ = get_env.step(unit_step)
    assert pytest.approx(reward, abs=50) == 10676.0, "Reward is not correct."


def test_reward_with_violations(get_env, reference_values, step_values):
    zero_step = step_values["zero_step"]
    zero_step["Cross-lapperLayersCount"] = 2.0
    zero_step["Needleloom1FeedPerStroke"] = 12.0
    get_env._objective_manager._config.output_bounds["TensileStrengthCD"]["lower"] = 1000
    reward, _, _, _ = get_env.step(zero_step)
    assert reward < 0, "Penalty has not been set."


# Test set 5: Test correctness of env setup with parallel execution
def test_step_parallel_processing(get_env, reference_values, step_values, config):
    config.parallel_execution = True
    factory = EnvironmentFactory(config)
    environment = factory.create_environment()
    environment.reset()

    unit_step = step_values["unit_step"]
    reference_setpoints = reference_values["reference_setpoints"]
    reward, state, _, _ = get_env.step(unit_step)

    for key, value in reference_setpoints.items():
        assert pytest.approx(value + UNIT_STEP) == state[key], (f"Key '{key}' has wrong value after unit step "
                                                                f"(Parallel execution).")
    environment.close()


# Test set 6: Test correctness of gym wrapper with action scaling

def test_gym_wrapper_action_transformation(get_env, reference_values, step_values, config):
    get_env._action_manager._config.dependent_variable_bounds["MassThroughput"]["upper"] = 10000
    gym_wrapper = GymWrapper(get_env, config.gym_setup, config.action_setup, config.env_setup)
    zero_step = step_values["zero_step"]
    reference_setpoints = reference_values["reference_setpoints"]

    keys = list(config.action_setup.initial_setpoints.keys())
    bounds = OmegaConf.to_container(config.action_setup.setpoint_bounds)
    step_as_array = transformations.dict_to_array(zero_step, keys, bounds, mode="inverse_tanh")
    observations, reward, _, _, _ = gym_wrapper.step(step_as_array)
    assert abs(reward) < 1, "Reward has not been scaled."

    for index, (key, value) in enumerate(reference_setpoints.items()):
        assert pytest.approx(value) == observations[index], (f"Key '{key}' has wrong value after zero step using "
                                                             f"gymwrapper ")

    far_away_step = step_as_array + 1000
    observations, reward, _, _, _ = gym_wrapper.step(far_away_step)
    for index, (key, value) in enumerate(config.action_setup.setpoint_bounds.items()):
        assert pytest.approx(value.upper, rel=0.05) == observations[index], (f"Key '{key}' has wrong value after far "
                                                                             f"away step using gymwrapper ")
    gym_wrapper.close()


def test_gym_wrapper_state_transformation(get_env, reference_values, step_values, config):
    get_env._action_manager._config.dependent_variable_bounds["MassThroughput"]["upper"] = 10000
    get_env._action_manager._config.dependent_variable_bounds["MassThroughput"]["lower"] = 0
    config.gym_setup.scale_observations = True
    gym_wrapper = GymWrapper(get_env, config.gym_setup, config.action_setup, config.env_setup)

    zero_step = step_values["zero_step"]
    keys = list(config.action_setup.initial_setpoints.keys())
    bounds = OmegaConf.to_container(config.action_setup.setpoint_bounds)
    step_as_array = transformations.dict_to_array(zero_step, keys, bounds)

    very_low_step = step_as_array - 1000

    observations, reward, _, _, _ = gym_wrapper.step(very_low_step)
    for index, value in enumerate(keys):
        assert pytest.approx(-1, abs=0.01) == observations[index], (f"Observation number '{index}' has wrong value "
                                                                    f"after very low step using gymwrapper ")

    far_away_step = step_as_array + 1000
    observations, reward, _, _, _ = gym_wrapper.step(far_away_step)
    for index, value in enumerate(keys):
        assert pytest.approx(1, abs=0.01) == observations[index], (f"Observation number '{index}' has wrong value"
                                                                   f" after far away step using gymwrapper")
    gym_wrapper.close()
