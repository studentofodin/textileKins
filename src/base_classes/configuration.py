# Delete this class and move its contents to the Configuration file


from src.abstract_base_class.configuration import AbstractConfiguration


class Configuration(AbstractConfiguration):
    @property
    def requirements(self) -> dict:
        return self._requirements

    @property
    def actorConstraints(self) -> dict:
        return self._actionParams

    @property
    def productionScenario(self) -> dict:
        return self._productionScenario

    @property
    def actionParams(self) -> dict:
        return self._actionParams

    @property
    def stateParams(self) -> dict:
        return self._stateParams

    @property
    def stepsUntilLabDataAvailable(self) -> int:
        return self._stepsUntilLabDataAvailable

    @property
    def observationParams(self) -> dict:
        return self._observationParams

    @property
    def maxSteps(self) -> int:
        return self._maxSteps

    @requirements.setter
    def requirements(self, requirements):
        if type(requirements) == dict:  # type check for dict
            if requirements:  # check if dictionary not empty
                self._requirements = requirements
            else:
                raise ValueError
        else:
            raise ValueError

    @actorConstraints.setter
    def actorConstraints(self, actorConstraints):
        if type(actorConstraints) == dict:  # type check for dict
            if actorConstraints:  # check if dictionary not empty
                self._actorConstraints = actorConstraints
            else:
                raise ValueError
        else:
            raise ValueError

    @productionScenario.setter
    def productionScenario(self, productionScenario):
        if type(productionScenario) == dict:  # type check for dict
            if productionScenario:  # check if dictionary not empty
                self._productionScenario = productionScenario
            else:
                raise ValueError
        else:
            raise ValueError

    @actionParams.setter
    def actionParams(self, actionParams):
        if type(actionParams) == dict:  # type check for dict
            if actionParams:  # check if dictionary not empty
                self._actionParams = actionParams
            else:
                raise ValueError
        else:
            raise ValueError

    @stateParams.setter
    def stateParams(self, stateParams):
        if type(stateParams) == dict:  # type check for dict
            if stateParams:  # check if dictionary not empty
                self._stateParams = stateParams
            else:
                raise ValueError
        else:
            raise ValueError

    @stepsUntilLabDataAvailable.setter
    def stepsUntilLabDataAvailable(self, stepsUntilLabDataAvailable):
        if type(stepsUntilLabDataAvailable) == int:  # type check for dict
            self._stepsUntilLabDataAvailable = stepsUntilLabDataAvailable
        else:
            raise ValueError

    @observationParams.setter
    def observationParams(self, observationParams):
        if type(observationParams) == dict:  # type check for dict
            if observationParams:  # check if dictionary not empty
                self._observationParams = observationParams
            else:
                raise ValueError
        else:
            raise ValueError

    @maxSteps.setter
    def maxSteps(self, maxSteps):
        if type(maxSteps) == int:  # type check for int
            if maxSteps > 0:  # check if steps not 0
                self._maxSteps = maxSteps
            else:
                raise ValueError
        else:
            raise ValueError

    def __init__(
        self,
        requirements,
        actorConstraints,
        productionScenario,
        actionParams,
        stateParams,
        stepsUntilLabDataAvailable,
        observationParams,
        maxSteps=200
    ):
        self.requirements = requirements
        self.actorConstraints = actorConstraints
        self.productionScenario = productionScenario
        self.actionParams = actionParams
        self.stateParams = stateParams
        self.stepsUntilLabDataAvailable = stepsUntilLabDataAvailable
        self.observationParams = observationParams
        self.maxSteps = maxSteps

    def readConfigFile(self, pathToFile):
        pass
    
    def __str__(self) -> str:
        print("Configuration Manager ----")
        print("Requirements : ", self.requirements)
        print("Actor Constraints : ", self.actorConstraints)
        print("Production Scenario : ",  self.productionScenario)
        print("Action Parameters : ", self.actionParams)
        print("State Parameters : ", self.stateParams)
        print("Steps Left : ", self.stepsUntilLabDataAvailable)
        print("Observation Parameters : ", self.observationParams)
        print("Maximum Steps for Training : ", self.maxSteps)
        return " "
