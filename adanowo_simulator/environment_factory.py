from omegaconf import DictConfig, OmegaConf

from adanowo_simulator.objective_manager import ObjectiveManager
from adanowo_simulator.action_manager import ActionManager
from adanowo_simulator.disturbance_manager import DisturbanceManager
from adanowo_simulator.output_manager import ParallelOutputManager, SequentialOutputManager
from adanowo_simulator.scenario_manager import ScenarioManager
from adanowo_simulator.experiment_tracker import WandBTracker, EmptyTracker
from adanowo_simulator.environment import Environment
from adanowo_simulator.objective_functions import baseline_objective, baseline_penalty


class EnvironmentFactory:
    def __init__(self, config: DictConfig):
        self.config = config

    def create_disturbance_manager(self):
        return DisturbanceManager(self.config.disturbance_setup)

    def create_action_manager(self):
        return ActionManager(self.config.action_setup, self.config.action_setup.actions_are_relative)

    def create_output_manager(self):
        # Decide whether to create a SequentialOutputManager or ParallelOutputManager
        if self.config.parallel_execution:
            return ParallelOutputManager(self.config.output_setup)
        else:
            return SequentialOutputManager(self.config.output_setup)

    def create_objective_manager(self):
        return ObjectiveManager(baseline_objective, baseline_penalty, self.config.objective_setup)

    def create_scenario_manager(self):
        return ScenarioManager(self.config.scenario_setup)

    def create_experiment_tracker(self):
        if self.config.tracking_enabled:
            return WandBTracker(self.config.wandb_settings, self.config)
        else:
            return EmptyTracker(OmegaConf.create(), OmegaConf.create())

    def create_environment(self):
        return Environment(
            self.config.env_setup,
            self.create_disturbance_manager(),
            self.create_action_manager(),
            self.create_output_manager(),
            self.create_objective_manager(),
            self.create_scenario_manager(),
            self.create_experiment_tracker()
        )
