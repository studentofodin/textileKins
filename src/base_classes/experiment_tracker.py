import numpy as np
import wandb as wb

from src.abstract_base_class.experiment_tracker import AbstractExperimentTracker


class ExperimentTracker(AbstractExperimentTracker):
    @property
    def metrics(self) -> dict:
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        if type(metrics) == dict:  # type check for dict
            if metrics:  # check if dict not empty
                self._metrics = metrics
            else:
                raise ValueError
        else:
            raise ValueError

    def plotReward(self, rewardValue) -> bool:
        return True

    def createMetricFromConfig(self, metricConfig):
        metrics = {}
        for metric in metricConfig:
            metrics[metric] = []
        return metrics

    def __init__(self, config):
        self.metrics = self.createMetricFromConfig(config.experimentTracker.metrics)
        wb.init(config = config)

    def __str__(self) -> str:
        print("Experiment Tracker ---- ")
        print("Metrics", self.metrics)
        return " "

    def log(self, reward, stateDict, observationDict):
        wb.log({"Reward":reward, **stateDict, **observationDict})

