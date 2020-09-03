"an infeed generator "
from typing import Dict

import tensorflow as tf
from absl import logging

import lm

from .base import Infeed, InfeedConfig


class TFRecordDatasetReaderConfig(InfeedConfig):
    producer: Dict


@lm.register_infeed("lm.infeeds.TFRecordDatasetReader")
class TFRecordDatasetReader(Infeed):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(TFRecordDatasetReaderConfig(**kwds)))

    # def __call__(self, params: Dict):
    #     producer = self.create_producer()

    #     batch_size = params["batch_size"]
    #     context_length = producer.context_length
    #     example_sequence_shape = tf.TensorShape((batch_size, context_length))

    #     dataset = tf.data.Dataset.from_generator(
    #         producer,
    #         output_types=(tf.int64, tf.int64),
    #         output_shapes=(example_sequence_shape, example_sequence_shape),
    #     )
    #     return dataset

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
