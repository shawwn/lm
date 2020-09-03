"Base infeed class"
import abc

import tensorflow as tf
from pydantic import BaseModel


class InfeedConfig(BaseModel):
    batch_size: int
    # dataset: Dict
    # max_sequence_length: int
    # file_pattern: Optional[str]


class Infeed(abc.ABC):
    """
    An infeed abstracts the operation of creating a stream of examples.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwds) -> tf.data.Dataset:
        """
        Configures and Allocates a tensorflow dataset
        """
