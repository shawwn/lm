"""
Sequence to sequence input configuration
"""
import tensorflow as tf
from absl import logging

import lm

from .base import Infeed, InfeedConfig


# @lm.register_infeed('Seq2SeqTFRecordInfeed', InfeedConfig)
class Seq2SeqTFRecordInfeed(Infeed):
    """Wrapper class that acts as the input_fn to TPUEstimator."""

    def __init__(self, config: InfeedConfig):
        self.config = config
        self.load_dataset(config.dataset)

    def load_dataset(self, dataset):
        self.dataset = lm.datasets.from_config(dataset)
        return self.dataset

    def __call__(self, params):
        """Input function which provides a single batch for train or eval."""
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # `tf.estimator.tpu.RunConfig` for details.
        batch_size = params["batch_size"]
        max_seq_length = params["max_sequence_length"]

        logging.info(
            "call Seq2SeqTFRecordDataset() with batch size {} and sequence length",
            batch_size,
            max_seq_length,
        )

        filenames = tf.io.gfile.glob(self.config.file_pattern)
        logging.info(
            "Found %s files matching %s" % (len(filenames), self.config.file_pattern)
        )
        if not filenames:
            raise ValueError("No matching files found")
        ds = tf.data.TFRecordDataset(filenames, buffer_size=64 * 1024 * 1024)
        keys = ["content", "target"]

        # Examples are already pre-processed
        def decode_example(serialized_example):
            """Return a dict of Tensors from a serialized tensorflow.Example."""
            decoded = tf.io.parse_example(
                serialized=[serialized_example],
                features={k: tf.VarLenFeature(tf.int64) for k in keys},
            )
            decoded = {k: v.values for k, v in decoded.items()}
            return decoded["content"], decoded["target"]

        ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return ds

    def eval(self, params):
        ds = self(params)
        ds = ds.batch(params["batch_size"], drop_remainder=False)
        return ds

    def train(self, params):
        ds = self(params)
        ds = ds.repeat()
        ds = ds.shuffle(1000)
        ds = ds.batch(params["batch_size"], drop_remainder=True)
        return ds
