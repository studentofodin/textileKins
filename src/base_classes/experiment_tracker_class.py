

from src.abstract_base_class.experiment_tracking import AbstractExperimentTracking


class ExperimentTrackingClass(AbstractExperimentTracking):

    @property
    def metric(self):
        return self._metric

    def plotReward(self, rewardValue) -> bool:
        return True

    def __init__(self, metric):
        self._metric = metric
