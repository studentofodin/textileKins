# adanowo_simulator: A Python Library for Simulating Nonwoven Production Environments 
This project is meant for testing and validationg smart production planning agents in a simulated nonwoven production environment.

<img src="Logo.png" alt="Project Logo" width="500"/>

## Description

This project serves as a simulation tool for nonwovens production, targeting the textile industry — a sector of ongoing relevance in Europe. 
Designed to overcome the challenges of incomplete optimization problem formulations and not reproducible results of many research works, the simulator serves as a robust platform for testing various agent algorithms for process optimization.

### Key Features:

- **Data-Driven**: Utilizes machine learning models trained on industrial data, alongside physically motivated models, for accurate simulation.
- **Modular and Adaptable**: Built using Python 3.10, the project is highly modular and allows for comprehensive and reproducible testing of different optimization approaches.
- **Economical & Technical**: Captures both the economic and technical aspects of production, thereby offering a 'simulation-first' approach for cheap testing before physical trials.
- **No Dynamics**: The simulator assumes a stationary process after all dynamic effects have alread decayed.

By integrating advanced data-driven methods, this simulation tool aims to reduce waste, increase profitability, and alleviate the workload of staff in the textile industry. 
The project is set to contribute to ongoing research in production process optimization.


## Installation

### Prerequisites:
- Python 3.10.4 or higher, lower than 3.11
- (optional) a fresh virtual environment
- [Poetry](https://python-poetry.org/docs/) is already preinstalled. 
Under Windows you might need to use pip to install poetry if the command from the website doesn't work.
- CUDA 11.7 (Optional but highly recommended for better performance)

### Steps:

1. Clone the repository:
   ```bash
   git clone https://git.rwth-aachen.de/ruben.kins/adanowo-simulator.git

2. Change to the project directory:
   ```bash
   cd adanowo-simulator

3. Install the required packages using Poetry:
   ```bash
   poetry install
   
4. Note: The Poetry command will install the project in editable mode. If you want to install the project as a regular, 
package, use the following command afterwards (specifying the version number):
   ```bash
   poetry build
   pip install .\dist\adanowo_simulator-x.x.x.tar.gz

## Usage
Once you've installed all prerequisites and dependencies, you're ready to run the project.

### Logging with WandB
 This project also uses [Weights and Biases](https://wandb.ai/site) for logging and visualization.
Make sure you have setup an account before so you can enter your credentials on the first run. 
Alternatively, you can disable logging or use a different logger by writing your own simple logger module.

### Configuration with Hydra
This project uses Hydra for configuration management. Configuration files can be found in the `config` folder. 
You can customize the run by modifying these configuration files or by passing parameters through the command line. 
For more information on how to use Hydra, refer to their [documentation](https://hydra.cc/docs/intro/).
The config folder will not be installed by poetry, its is meant to be used by the calling script. 
Use the existing configuration as a template for your own tests.

### Example Usage
A detailed example can be found in `./examples/example.py`. 
The example script demonstrates how to use the `EnvironmentFactory` and `GymWrapper` to initialize and step through an 
environment.
A simple random agent is used in the example. For adapting the code to arbitrary agents, 
you need to write your own wrapper file.


## Explanantion of the Environment class
The environment class simulates the reaction of a physical nonwovens production process to different setpoint values. 
The goal is to maximize the objective value, which is calculated from the process contribution margin while not 
violating any product quality or actor constraints. Domain randomization via scenarios ensures the agent can train on 
different production contexts and is robust to changes in the process behavior and product requirements.


The Environment class handles all top-level interactions in this project. 
It is meant to be wrapped by another class to ensure compatibility with all kinds of different agent architectures.
From an outside point of view, an environment receives **actions** from an **agent**, processes them, 
and returns **observations** as well as an **objective/reward value**. This is called a **step**.


After receiving an action with the same length as the setpoints, the environment checks wether stationary, a priori 
known actor constraints or other dependent variables have been violated. If so, the action is not performed and a 
penalty is returned. Next, all the relevant states (setpoints, disturbances, dependent variables) are updated and passed 
to the probabilistic process simulation models. The models simulate the process output behavior based on the process 
state. Probabilistic realizations of important economic and technical production outcomes are now available. Next, the 
objective function is evaluated. The objective is mostly economically motivated, but is also checked against output 
bounds to ensure all quality constraints are satisfied. If they are violated, a penalty is returned instead of the 
objective value. This simulates qualit rejects that the customer will not accept. The objective value is returned to the 
agent, along with the new process state and other context (process outputs, quality bounds).


### Naming scheme

- **Disturbances**: Variables that cannot be manipulated.
- **Setpoints**: Variables that can be manipulated to maximize the objective value. 
They are calculated from the actions either in an absolute or relative manner.
  - Absolute manner: setpoints(t) = actions(t)
  - Relative manner: setpoints(t) = setpoints(t-1) + actions(t)
- **Dependent variables**: Important variables calculated from the setpoints and disturbances. 
They might be calulated for checking constraints or simply as helper functions with outputs used elsewhere.
- **State**: Combination of disturbances, setpoints, and dependent variables.
- **Outputs**: Variables that exhibit probabilistic behavior and are calculated from the state using models.
- **Objective/reward value**: A value that is calculated from the outputs and state and is to be maximized.
- **Output bounds**: Bounds that the outputs must not exceed, or a penalty is returned. This simulates the quality 
characteristics of the produced nonwoven that the customer demands.
- **Scenario**: Changes the environment behavior in a deterministic or random manner. This can be used to simulate changing
quality requirements, disturbances or process behaviors.

> **Note**: Assume variable \`y\` depends on a variable \`x\`. We use the term **calculation** if a value for 
> \`y\` is directly returned given \`x\`.

### Members of the class

- \`disturbance_manager\`: Gets the disturbances and returns them.
- \`action_manager\`: Checks whether the setpoint and dependent variable constraints are satisfied.
- \`output_manager\`: Gets the outputs using the models and returns the outputs.
- \`objective_manager\`: Checks if the output bounds are satisfied and returns the value of the objective function.
- \`scenario_manager\`: Implements domain randomization by changing the configuration of other members (constraints, 
disturbances or quality bounds).
- \`experiment_tracker\`: Tracks the results from a sequence of steps.

### API Methods

- \`step\`: Updates the environment with actions, returning observations and an objective value.
- \`reset\`: Resets the environment to initial variable values.
- \`close\`: Closes the environment, shutting down any background processes.

> **Note**: The API mainly works with dictionaries, specifically \`dict[str, float]\`, 
> whose keys represent names of numerical variables.


## Modifications
The environment in this project is designed to be highly modular, allowing you to easily extend or modify its functionalities. 
Abstract base classes with detailed docstrings are provided to serve as a template for creating new modules.


To write your own module, start by examining the abstract base classes provided in the `abstract_base_classes` directory. 
Each abstract base class contains thorough docstrings explaining its purpose and how to implement its methods.


## Reference to Academic Paper

This software project is part of the broader research initiative "adanowo".
A comprehensive research paper discussing the methodologies, models, and results has been published. 
We strongly recommend reading the paper for a more in-depth understanding of the algorithms and techniques implemented here.

### Citation

You can cite this project and the associated paper as follows:

#### APA Style
*To be updated upon publication*

#### IEEE Style
*To be updated upon publication*

For more detailed information, please refer to the research paper or contact the author under ruben.kins@rwth-aachen.de.
