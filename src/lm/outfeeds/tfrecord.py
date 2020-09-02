"TFRecord output feed"
import tensorflow as tf
from absl import app, logging
from tensorflow.compat import v1
from pydantic import BaseModel
from tqdm import auto as tqdm
from typing import Dict, Callable

class TFRecordOutFeedConfig(BaseModel):
    output_location:str
    example_producer: Callable
    n_samples:int
    compress:bool = False

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def example_proto(example):
    # Helper function to avoid code duplication, writes the data as an example to the file and increments i
    content, target = example
    feature = {"content": _int64_feature(content), "target": _int64_feature(target)}
    return tf.train.Example(features=tf.train.Features(feature=feature))


class TFRecordOutFeed:

    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(TFRecordOutFeedConfig(**kwds).dict()) 

    def __call__(self):
        with tf.io.TFRecordWriter(self.output_location) as w:
            it = iter(self.example_producer(seq))
            for _ in tqdm.tqdm(range(self.n_samples)):
                batch_ex = next(it)
                for c, t in zip(batch_ex[0], batch_ex[1]):
                    proto = example_proto((c, t))
                    w.write(proto.SerializeToString())
