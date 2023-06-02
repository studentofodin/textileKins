from abc import ABC, abstractmethod
from omegaconf import DictConfig


class AbstractScenarioManager(ABC):
    @property
    @abstractmethod
    def config(self) -> DictConfig:
        pass

    @config.setter
    @abstractmethod
    def config(self, c):
        pass

    def update_model_allocation(self, step_index: int, output_models_config: DictConfig) -> tuple[DictConfig, list[str]]:
        """
        change the output_models_config according to own config.
        return the changed output_models_config and the output names which the model entry in the output_models_config
        was changed for.
        """
        pass

    def update_requirements(self, step_index: int, requirements_config: DictConfig) -> DictConfig:
        """
        change and return the requirements_config according to own config.
        """
        pass

    def update_disturbances(self, step_index: int, disturbance_config: DictConfig) -> DictConfig:
        """
        change and return the disturbance_config according to own config.
        """
        pass

    def reset(self) -> None:
        """
        reset to initial values.
        """
        pass
