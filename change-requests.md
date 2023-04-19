# Changelog
### v_0.2
- merged changes from Luis
- switched from Gpy to Gpytorch (Gpy was deprecated)

# Change-requests
- Add the requirements to the logging because they might change during our run
- To Do Ruben: Implement OperatorFlag, a flag that gets triggered when the process is being operated in  astate where the models are not trained. This is meant to simulate "risky settings".
- TO Do Ruben: Check for Cuda GPU
- implement logging using the logger module: https://realpython.com/python-logging/
  - events from the scenario manager
  - events from the TrainingEnvironment class: init complete, reset complete
