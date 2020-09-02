"Base Task Class"
from abc import ABC, abstractmethod

from pydantic import BaseModel

class BaseTaskDatasetConfig(BaseModel):
    kind:str

class BaseTaskConfig(BaseModel):
    kind: str
    description: str

class BaseTask(ABC):

    @abstractmethod    
    def infeed(self):
        pass
