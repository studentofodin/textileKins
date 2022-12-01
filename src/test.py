from base_classes.scenario_manager_class import ScenarioManager
from base_classes.configuration_class import Configuration
from base_classes.experiment_tracker_class import ExperimentTracker
from base_classes.env_wrapper_class import ITAEnvWrapper

def main():
    scenarioManager = ScenarioManager({'key1':1,'key2':2}, {'key3':3,'key4':4})
    print(scenarioManager)
    print(scenarioManager.disturbanceSetting)
    print(scenarioManager.fibreSetting)
    scenarioManager.disturbanceSetting={'key5':5,'key6':6}
    scenarioManager.fibreSetting={'key7':7,'key8':8}
    print(scenarioManager.disturbanceSetting)
    print(scenarioManager.fibreSetting)
    print("Scenario Manager Tested!")

    configuration = Configuration(requirements={'req1':123,'req2':321},actorConstraints={'constraint1':4,'constraint2':5}, productionScenario={'pS1':2,'pS2':5}, actionParams={'ap1':4,'ap2':5}, stateParams={'sP1':3,'sP2':4}, stepsUntilLabDataAvailable=100, observationParams={'oP1':4,'oP2':9})
    print(configuration)
    configuration.requirements={'New Req1':222,'New Req2':111}
    print(configuration.requirements)
    print("Configuration Tested!")

    experimentTracker = ExperimentTracker(['Reward', 'Weight Per Area', 'Metric3'])
    print(experimentTracker)
    print(experimentTracker.metrics)
    print("Experiment Tracker Tested!")

    env = ITAEnvWrapper()
main()