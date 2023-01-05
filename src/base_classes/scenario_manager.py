from omegaconf import OmegaConf
from src.abstract_base_class.scenario_manager import AbstractScenarioManager


class ScenarioManager(AbstractScenarioManager):
    @property
    def disturbanceSetting(self) -> dict:
        return self._disturbanceSetting

    @property
    def fibreSetting(self) -> dict:
        return self._fibreSetting

    @disturbanceSetting.setter
    def disturbanceSetting(self, disturbanceSetting):
        if type(disturbanceSetting) == dict:  # type check for dict
            if disturbanceSetting:  # check if dictionary not empty
                self._disturbanceSetting = disturbanceSetting
            else:
                raise ValueError
        else:
            raise ValueError

    @fibreSetting.setter
    def fibreSetting(self, fibreSetting):
        if type(fibreSetting) == dict:  # type check for dict
            if fibreSetting:  # check if dictionary not empty
                self._fibreSetting = fibreSetting
            else:
                raise ValueError
        else:
            raise ValueError

    def __init__(self, disturbanceSetting, fibreSetting):
        self.disturbanceSetting = dict(disturbanceSetting)
        self.fibreSetting = dict(fibreSetting)

    def __str__(self):
        print("Scenario Manager ---")
        print("Disturbance Setting : ")
        print(self.disturbanceSetting)
        print("Fibre Setting : ")
        print(self.fibreSetting)
        return ""