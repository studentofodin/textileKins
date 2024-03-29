import numpy as np
import warnings

from omegaconf import OmegaConf
from gymnasium import Env, spaces
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import register
from omegaconf import DictConfig

from adanowo_simulator.abstract_base_classes.environment import AbstractEnvironment
from adanowo_simulator import transformations


class GymWrapper(Env):

    def __init__(self, environment: AbstractEnvironment, config: DictConfig, action_config: DictConfig,
                 env_config: DictConfig):
        self._environment: AbstractEnvironment = environment
        self._config: DictConfig = config.copy()
        self._action_config = action_config.copy()
        self._env_config = env_config
        self._action_space: spaces.Box = spaces.Box(low=0, high=0)
        self._observation_space: spaces.Box = spaces.Box(low=0, high=0)

    @property
    def environment(self) -> AbstractEnvironment:
        return self._environment

    @property
    def action_space(self) -> spaces.Box:
        warnings.warn("The action space is not yet implemented. You might get unwanted behaviors.")
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        warnings.warn("The observation space is not yet implemented. You might get unwanted behaviors.")
        return self._observation_space

    def step(self, actions_array: np.array) -> tuple[np.array, float, bool, bool, dict]:
        if self._config.scale_and_constrain_actions and self._action_config.actions_are_relative:
            raise NotImplementedError("Scaling is not supported for relative actions.")

        # transform actions
        setpoint_keys = list(self._env_config.used_setpoints)
        if self._config.scale_and_constrain_actions:
            scales = OmegaConf.to_container(self._action_config.setpoint_bounds)
            action = transformations.array_to_dict(actions_array, setpoint_keys, bounds_for_scaling=scales)
        else:
            action = transformations.array_to_dict(actions_array, setpoint_keys)

        # perform step
        reward, state, outputs, quality_bounds = self._environment.step(action)
        # TODO: Implement quality bounds as optional part of observations

        # post-processing of env returns
        observations = self._compile_observations(state, outputs)
        if self._config.scale_rewards:
            reward = reward / self._config.max_reward
        return observations, reward, False, False, dict()

    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        super().reset(seed=seed)
        reward, state, outputs, quality_bounds = self._environment.reset()
        observations = self._compile_observations(state, outputs)
        return observations, dict()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self) -> None:
        self._environment.close()

    def _compile_observations(self, state, outputs):
        # tranform state variables
        setpoint_list = list(self._env_config.used_setpoints)
        setpoint_states = {key: state[key] for key in setpoint_list}
        if self._config.scale_observations:
            bounds = OmegaConf.to_container(self._action_config.setpoint_bounds)
            observations = transformations.dict_to_array(setpoint_states, setpoint_list, bounds_for_scaling=bounds)
        else:
            observations = transformations.dict_to_array(setpoint_states, setpoint_list)

        if not self._config.return_process_outputs:  # do not concatenate process outputs
            return observations

        # concatenate process outputs
        output_keys = list(self._env_config.used_outputs) + list(self._env_config.used_dependent_variable_setpoints)
        if self._config.scale_observations:
            raise NotImplementedError("The output bounds need to be updated in the config.")
            # bounds = self._config.process_output_bounds
            # observations_outputs = transformations.dict_to_array(outputs, keys, bounds)
        else:
            observations_outputs = transformations.dict_to_array(outputs, output_keys)
        observations = np.concatenate(
            (
                observations,
                observations_outputs
            )
        )
        return observations


register(
    id='adaNowo-simulator-v1',
    entry_point='src.base_classes.gym_wrapper.GymWrapper'
)
