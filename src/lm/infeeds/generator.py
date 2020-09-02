"Module to define a generator based infeed"
from .base import Infeed
from tensorflow.compat import v1
import tensorflow as tf

from absl import logging

class InfeedGeneratorWrapper(Infeed):
    def __init__(self, infeed:Infeed):
        super().__init__()
        self.infeed = infeed

    def __call__(self):
        with v1.Session(graph=tf.Graph()) as sess:
            ds = self.infeed({"batch_size": 8})

            it = ds.make_one_shot_iterator()
            example = it.get_next()
            while True:
                try:
                    yield sess.run(example)
                except tf.errors.OutOfRangeError:
                    logging.error("unexpected end of infinite dataset",)

