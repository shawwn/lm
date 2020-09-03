"Module to define a generator based infeed"
from typing import Dict, Iterable

import tensorflow as tf
from absl import logging
from tensorflow.compat import v1

from lm.infeeds.base import Infeed


def consume(infeed: Infeed, params: Dict) -> Iterable:
    "consumes an infeed"
    with v1.Session(graph=tf.Graph()) as sess:
        ds = infeed(params)
        it = ds.make_one_shot_iterator()
        example = it.get_next()
        while True:
            try:
                yield sess.run(example)
            except tf.errors.OutOfRangeError:
                logging.error("unexpected end of infinite dataset")
