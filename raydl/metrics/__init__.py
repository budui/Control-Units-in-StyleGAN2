from ignite.engine.events import Events
from ignite.metrics import MetricUsage

from .common import Collector
from .generation import FID


class LifeWise(MetricUsage):
    usage_name: str = "life_wise"

    def __init__(self) -> None:
        super(LifeWise, self).__init__(
            started=Events.STARTED,
            completed=Events.COMPLETED,
            iteration_completed=Events.ITERATION_COMPLETED,
        )
