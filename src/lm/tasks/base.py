"Base Task Class"
from abc import ABC, abstractmethod

from pydantic import BaseModel

from lm.infeeds import InfeedConfig


class BaseTaskDatasetConfig(BaseModel):
    kind: str


class BaseTaskConfig(BaseModel):
    kind: str
    description: str
    infeed: InfeedConfig


class BaseTask(ABC):
    @abstractmethod
    def build_infeed(self):
        pass
