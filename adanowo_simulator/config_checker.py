from omegaconf import DictConfig, OmegaConf
from numbers import Number


class ConfigChecker:

    def __init__(self, config: DictConfig):
        self._config = config.copy()
        self._model_wrapper = None

    def check_config(self):
        """
        apply self._check_type everywhere!
        """
        used_controls = self._config.env_setup.used_controls
        used_disturbances = self._config.env_setup.used_controls
        used_outputs = self._config.env_setup.used_disturbances

        self._check_used(self._config.env_setup.used_controls, "used_controls")
        self._check_used(self._config.env_setup.used_disturbances, "used_disturbances")
        self._check_used(self._config.env_setup.used_outputs, "used_outputs")

        # self._check_output_models()
        # self._check_action_space(used_controls)
        # self._check_observation_space(used_outputs)
        # self._check_initial_controls(used_controls)
        # self._check_disturbances(used_disturbances)
        # self._check_safety(used_controls)
        # self._check_penalty()
        # self._check_requirements(used_outputs)
        # self._check_output_models(used_outputs)

        print("Config is fine.")

    def _check_used(self, used_names: list[str], config_name: str):

        for name in used_names:
            self._check_type(name, f"{name}", )
            if not isinstance(name, str):
                raise Exception(f"{name} in {config_name} is not of {str}.")

        duplicate = self._first_duplicate(used_names)
        if duplicate != "":
            raise Exception(f"{duplicate} occurs several times in {config_name}.")


    def _check_space(self, space_config: DictConfig, config_name: str):

        for name, limits in space_config.items():
            if not isinstance(limits.low, Number):
                raise Exception(f"{config_name}/{name}/low is not of {Number}.")
            if not isinstance(limits.high, Number):
                raise Exception(f"{config_name}/{name}/high is not of {Number}.")
            if limits.low > limits.high:
                raise Exception(f"{config_name}/{name} has larger low than high value.")

        config_keys_list = list(config.keys())
        if used_controls != config_keys_list:
            used_controls_set = set(used_controls)
            config_keys_set = set(config.keys())
            diff = used_controls_set.difference(config_keys_set)
            if diff:
                raise Exception(f"{diff} of used_controls are not in keys of action_space.")
            diff = config_keys_set.difference(used_controls_set)
            if diff:
                raise Exception(f"keys {diff} of action_space are not in used_controls.")
            duplicate = self._first_duplicate(config_keys_list)
            if duplicate != "":
                raise Exception(f"{duplicate} occurs several times in action space.")
            raise Exception(f"order of used_controls and keys of action space are different.")



    def _check_action_space(self, used_controls: list[str]) -> None:
        config = self._config.env_setup.action_space

        for name, limits in config.items():
            if not isinstance(limits.low, numbers.Number):
                raise Exception(f"action_space/{name}/low is not of type numeric.")
            if not isinstance(limits.high, numbers.Number):
                raise Exception(f"action_space/{name}/high is not of type numeric.")
            if limits.low > limits.high:
                raise Exception(f"action_space/{name} has larger low than high value.")

        config_keys_list = list(config.keys())
        if used_controls != config_keys_list:
            used_controls_set = set(used_controls)
            config_keys_set = set(config.keys())
            diff = used_controls_set.difference(config_keys_set)
            if diff:
                raise Exception(f"{diff} of used_controls are not in keys of action_space.")
            diff = config_keys_set.difference(used_controls_set)
            if diff:
                raise Exception(f"keys {diff} of action_space are not in used_controls.")
            duplicate = self._first_duplicate(config_keys_list)
            if duplicate != "":
                raise Exception(f"{duplicate} occurs several times in action space.")
            raise Exception(f"order of used_controls and keys of action space are different.")


    def _check_observation_space(self, used_outputs: list[str]) -> None:
        observation_space_config = self._config.env_setup.observation_space

        for observation_name, config in observation_space_config.items():
            if not isinstance(config.low, numbers.Number):
                raise Exception(f"observation_space/{observation_name}/low is not of type numeric.")
            if not isinstance(config.high, numbers.Number):
                raise Exception(f"observation_space/{observation_name}/high is not of type numeric.")
            if config.low > config.high:
                raise Exception(f"observation_space/{observation_name} has larger low than high value.")

        config_keys_list = list(action_space_config.keys())
        if used_controls != config_keys_list:
            used_controls_set = set(used_controls)
            config_keys_set = set(action_space_config.keys())
            diff = used_controls_set.difference(config_keys_set)
            if diff:
                raise Exception(f"{diff} of used_controls are not in keys of action_space.")
            diff = config_keys_set.difference(used_controls_set)
            if diff:
                raise Exception(f"keys {diff} of action_space are not in used_controls.")
            duplicate = self._first_duplicate(config_keys_list)
            if duplicate != "":
                raise Exception(f"{duplicate} occurs several times in action space.")
            raise Exception(f"order of used_controls and keys of action space are different.")



    def _check_initial_controls(self, used_controls: list[str]) -> None:
        initial_controls_config = self._config.process_setup.initial_controls

        for control_name in initial_controls_config.keys():
            if not isinstance(initial_controls_config[control_name], numbers.Number):
                raise Exception(f"initial_controls/{control_name} is not of type numeric.")

        if used_controls != list(initial_controls_config.keys()):
            raise Exception("used_outputs and the keys in initial_controls are not equal.")


    def _check_disturbances(self, used_disturbances: list[str]) -> None:
        disturbances_config = self._config.process_setup.disturbances

        for disturbance_name in disturbances_config.keys():
            if not isinstance(disturbances_config[disturbance_name], numbers.Number):
                raise Exception(f"disturbances/{disturbance_name} is not of type numeric.")

        if used_disturbances != list(disturbances_config.keys()):
            raise Exception("used_disturbances and the keys in disturbances are not equal.")

    def _check_safety(self, used_controls: list[str]) -> None:
        safety_config = self._config.process_setup.safety

        for control_name, config in safety_config.simple_control_bounds.items():
            if not isinstance(config.lower, numbers.Number):
                raise Exception(f"safety/simple_control_bounds/{control_name}/lower is not of type numeric.")
            if not isinstance(config.upper, numbers.Number):
                raise Exception(f"safety/simple_control_bounds/{control_name}/upper is not of type numeric.")
            if config.lower > config.upper:
                raise Exception(f"safety/{control_name} has larger lower than upper value.")

        for constraint_name, value in safety_config.complex_constraints.items():
            if not isinstance(value, numbers.Number):
                raise Exception(f"safety/complex_constraints/{constraint_name} is not of type numeric.")

        duplicate = self._first_duplicate(safety_config.simple_control_bounds.keys())
        if duplicate != "":
            raise Exception(f"{duplicate} occurs several times in safety_config/simple_control_bounds.")
        duplicate = self._first_duplicate(safety_config.complex_constraints.keys())
        if duplicate != "":
            raise Exception(f"{duplicate} occurs several times in safety_config/complex_constraints.")




        if len(set(safety_config.simple_control_bounds.keys())) < len(safety_config.simple_control_bounds.keys()):
            raise Exception("At least one element in the keys of safety/simple_control_bounds occurs several times.")
        if len(set(safety_config.complex_constraints.keys())) < len(safety_config.complex_constraints.keys()):
            raise Exception("At least one element in the keys of safety/complex_constraints occurs several times.")
        if not set(safety_config.simple_control_bounds.keys()).issubset(set(used_controls)):
            raise Exception("The keys of safety/simple_control_bounds are not a subset of used_controls.")


    def _check_penalty(self):
        penalty = self._config.product_setup.penalty
        if not isinstance(penalty, numbers.Number):
            raise Exception("penalty is not of type numeric.")
        if penalty < 0:
            raise Exception("penalty is smaller than 0.")


    def _check_requirements(self, used_outputs: list[str]) -> None:
        reqs_config = self._config.product_setup.requirements
        
        if len(set(reqs_config.simple_óutput_bounds.lower.keys())) < len(reqs_config.simple_output_bounds.lower.keys()):
            raise Exception("At least one element in the keys of requirements/simple_output_bounds/lower occurs several times.")
        if len(set(reqs_config.simple_óutput_bounds.lower.keys())) < len(reqs_config.simple_output_bounds.lower.keys()):
            raise Exception("At least one element in the keys of requirements/simple_output_bounds/lower occurs several times.")
        if len(set(safety_config.complex_constraints.keys())) < len(safety_config.complex_constraints.keys()):
            raise Exception("At least one element in the keys of safety/complex_constraints occurs several times.")
        if not set(safety_config.simple_control_bounds.keys()).issubset(set(used_controls)):
            raise Exception("The keys of safety/simple_control_bounds are not a subset of used_controls.")



    def _check_experiment_tracker(self):
        pass




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


    def _check_type(self, variable, variable_name: str, target_type: type) -> None:
        if not isinstance(variable, target_type):
            raise Exception(f"{variable_name} is not of {target_type}")

    def _first_duplicate(self, lst: list[str]) -> str:
        visited = set()
        for element in lst:
            if element in visited:
                return element
            visited.add(element)
        return ""









