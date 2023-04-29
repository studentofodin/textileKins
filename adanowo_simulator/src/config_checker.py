from omegaconf import DictConfig, OmegaConf
import numbers

from src.model_wrapper import ModelWrapper


class ConfigChecker:

    def __init__(self, config: DictConfig):
        self._config = config.copy()
        self._model_wrapper = None

    def check_config(self):
        used_controls = self._check_used_controls()
        used_disturbances = self._check_used_disturbances()
        used_outputs = self._check_used_outputs()
        self._check_action_space(used_controls)
        self._check_observation_space(used_outputs)
        self._check_initial_controls(used_controls)
        self._check_disturbances(used_disturbances)
        self._check_safety(used_controls)
        # self._check_requirements(used_outputs)
        # self._check_output_models(used_outputs)

        print("Config is fine.")


    def _check_used_controls(self) -> list[str]:
        used_controls = list(self._config.env_setup.used_controls)
        if len(set(used_controls)) < len(used_controls):
            raise Exception("At least one element in used_controls occurs several times.")
        for control_name in used_controls:
            if not isinstance(control_name, str):
                raise Exception(f"{control_name} in used_controls is not of type str.")
        return used_controls


    def _check_used_disturbances(self) -> list[str]:
        used_disturbances = list(self._config.env_setup.used_disturbances)
        if len(set(used_disturbances)) < len(used_disturbances):
            raise Exception("At least one element in used_disturbances occurs several times.")
        for disturbance_name in used_disturbances:
            if not isinstance(disturbance_name, str):
                raise Exception(f"{disturbance_name} in used_disturbances is not of type str.")
        return used_disturbances


    def _check_used_outputs(self) -> list[str]:
        used_outputs = list(self._config.env_setup.used_outputs)
        if len(set(used_outputs)) < len(used_outputs):
            raise Exception("At least one element in used_outputs occurs several times.")
        for output_name in used_outputs:
            if not isinstance(output_name, str):
                raise Exception(f"{output_name} in used_outputs is not of type str.")
        return used_outputs


    def _check_action_space(self, used_controls: list[str]) -> None:
        action_space_config = self._config.env_setup.action_space
        if used_controls != list(action_space_config.keys()):
            raise Exception("used_controls and the keys in action_space are not equal.")
        for action_name, config in action_space_config.items():
            if not isinstance(config.low, numbers.Number):
                raise Exception(f"action_space/{action_name}/low is not of type numeric.")
            if not isinstance(config.high, numbers.Number):
                raise Exception(f"action_space/{action_name}/high is not of type numeric.")
            if config.low > config.high:
                raise Exception(f"action_space/{action_name} has larger low than high value.")


    def _check_observation_space(self, used_outputs: list[str]) -> None:
        observation_space_config = self._config.env_setup.observation_space
        if used_outputs != list(observation_space_config.keys()):
            raise Exception("used_outputs and the keys in observation_space are not equal.")
        for observation_name, config in observation_space_config.items():
            if not isinstance(config.low, numbers.Number):
                raise Exception(f"observation_space/{observation_name}/low is not of type numeric.")
            if not isinstance(config.high, numbers.Number):
                raise Exception(f"observation_space/{observation_name}/high is not of type numeric.")
            if config.low > config.high:
                raise Exception(f"observation_space/{observation_name} has larger low than high value.")


    def _check_initial_controls(self, used_controls: list[str]) -> None:
        initial_controls_config = self._config.process_setup.initial_controls
        if used_controls != list(initial_controls_config.keys()):
            raise Exception("used_outputs and the keys in initial_controls are not equal.")
        for control_name in initial_controls_config.keys():
            if not isinstance(initial_controls_config[control_name], numbers.Number):
                raise Exception(f"initial_controls/{control_name} is not of type numeric.")


    def _check_disturbances(self, used_disturbances: list[str]) -> None:
        disturbances_config = self._config.process_setup.disturbances
        if used_disturbances != list(disturbances_config.keys()):
            raise Exception("used_disturbances and the keys in disturbances are not equal.")
        for disturbance_name in disturbances_config.keys():
            if not isinstance(disturbances_config[disturbance_name], numbers.Number):
                raise Exception(f"disturbances/{disturbance_name} is not of type numeric.")


    def _check_safety(self, used_controls: list[str]) -> None:
        safety_config = self._config.process_setup.safety

        if len(set(safety_config.simple_control_bounds.keys())) < len(safety_config.simple_control_bounds.keys()):
            raise Exception("At least on element in the keys of safety/simple_control_bounds occurs several times.")
        if len(set(safety_config.complex_constraints.keys())) < len(safety_config.complex_constraints.keys()):
            raise Exception("At least on element in the keys of safety/complex_constraints occurs several times.")
        if not set(safety_config.simple_control_bounds.keys()).issubset(set(used_controls)):
            raise Exception("The keys of safety/simple_control_bounds are not a subset of used_controls.")

        for control_name, config in safety_config.simple_control_bounds.items():
            if not isinstance(config.lower, numbers.Number):
                raise Exception(f"safety/simple_control_bounds/{control_name}/lower is not of type numeric.")
            if not isinstance(config.upper, numbers.Number):
                raise Exception(f"safety/simple_control_bounds/{control_name}/upper is not of type numeric.")
            if config.lower > config.upper:
                raise Exception(f"safety/{control_name} has larger lower than upper value.")


    # def _check_requirements(self, used_outputs: list[str]) -> None:
    #
    #
    #
    # def _check_output_models(self, used_outputs: list[str]) -> None:
    #     output_models_config = self._config.env_setupt.output_models
    #     if used_outputs != list(output_models_config.keys()):
    #         raise Exception("used_outputs and the keys in output_models are not equal.")
    #     for model_name in output_models_config.values():
    #         assert isinstance(model_name, str), f"output_models/{model_name} is not of type str."
    #     self._model_wrapper = ModelWrapper(OmegaConf.merge({"path_to_models": self._config.env_setup.path_to_models},
    #                                                        {"output_models": self._config.env_setup.output_models}))
    #     inputs = dict(self._config.process_setup.initial_controls) | dict(self._config.process_setup.disturbances)
    #     self._model_wrapper.get_outputs(inputs)








