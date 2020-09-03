"an infeed generator "
from typing import Dict

import tensorflow as tf

import lm

from .base import Infeed, InfeedConfig


class ExampleGeneratorConfig(InfeedConfig):
    producer: Dict


@lm.register_infeed("lm.infeeds.ExampleGenerator")
class ExampleGenerator(Infeed):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(ExampleGeneratorConfig(**kwds)))

    def create_producer(self):
        return self.get_dataset(self.producer)

    def __call__(self, params: Dict):
        producer = self.create_producer()

        batch_size = params["batch_size"]
        context_length = producer.context_length
        example_sequence_shape = tf.TensorShape((batch_size, context_length))

        dataset = tf.data.Dataset.from_generator(
            producer,
            output_types=(tf.int64, tf.int64),
            output_shapes=(example_sequence_shape, example_sequence_shape),
        )
        return dataset
