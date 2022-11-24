from base_classes.scenario_manager_class import ScenarioManager

def main():
    scenarioManager = ScenarioManager({'key1':1,'key2':2}, {'key3':3,'key4':4})
    print(scenarioManager)
    print(scenarioManager.disturbanceSetting)
    print(scenarioManager.fibreSetting)
    scenarioManager.disturbanceSetting={'key5':5,'key6':6}
    scenarioManager.fibreSetting={'key7':7,'key8':8}
    print(scenarioManager.disturbanceSetting)
    print(scenarioManager.fibreSetting)
    print("Scenario Manager Testedddddd")


main()