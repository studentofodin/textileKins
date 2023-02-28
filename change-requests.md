# Changelog
### v_0.1
- added folder top level for all installation files
- switched to poetry for easier dependency management
- switched from gym to gymnasium (gym is discontinued)
- restructured the config files
- removed old and unused artifacts
- restructured files
- Cleaned up the git branches to just:
  - main
  - review-ruben
  - dev-luis
  - dev-amrita
  - dev-tilman

# Change-requests
- Remove the ActionType from the config and make it an argument for instantiating the env 
- Implement the Scenario manager Module that allows us to simulate changing requirements, changing disturbances and changing system behavior during the experiment.
  - For that to work, we need to be able to change the Requirements and disturbances during the experiment. Here's how the config might look like:
    - Requirements
      - Tensile_strength
        - timestep: 1000
        - setting: 40.6
    - Disturbances
      - layersCount
        - timestep: 1000
        - setting: random
        - std: 40
    - Models
      - model1_name
        - timestep: 10000
        - new_model: model2_name
  - For disturbances there is a random setting shown: It means that every 1000 time steps the requirement changes randomly with a normal distribution around the original setting with a standard deviation of 40.
  - Also shown is a suggestion about how to define the changing of system behavior by loading a different model.
- Add the requirements to the logging because they might change during our run
- To Do Ruben: Implement OperatorFlag, a flag that gets triggered when the process is being operated in  astate where the models are not trained. This is meant to simulate "risky settings".
- To Do Ruben: Do we have models that take other model outputs in a recursive manner?
- implement logging using the logger module: https://realpython.com/python-logging/
  - events from the scenario manager
  - events from the TrainingEnvironment class: init complete, reset complete
# Questions
- Can the action and observation spaces be moved to the gym class?

- remove properties from machines
- check wandb for any errors I might have missed
- update git