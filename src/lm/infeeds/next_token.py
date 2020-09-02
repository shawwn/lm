from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from mesh_tensorflow import transformer
from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from tensorflow.python.platform import tf_logging as logging
from tokenizers import Tokenizer

import datasets
from encoders import encode


class Seq2SeqGeneratorConfig(BaseModel):
    context_length: int
    vocab_size: int


class RandomTokenGeneratorConfig(Seq2SeqGeneratorConfig):
    seed: int = 1337


class AddNSequenceGeneratorConfig(RandomTokenGeneratorConfig):
    n: int = 1


class AddNSequenceGenerator:
    """Generates Seq2Seq that are added by N"""

    def __init__(self, config: AddNSequenceGeneratorConfig):
        super().__init__()
        self.config = config
        assert self.config.context_length >= (
            3 + 1
        )  # 4 for the tokens and at least one number is needed

    def __call__(self, params):
        batch_size = params["batch_size"]
        vocab_size = self.config.vocab_size
        context_length = self.config.context_length
        np.random.seed(self.config.seed)

        def _generate():
            while True:
                # special tokens
                shape = (batch_size, 1)
                pad = np.full(shape, 0)  # pad token
                eos = np.full(shape, 1)  # end of sentence token
                bos = np.full(shape, 2)  # begin of sentence token
                num_special_tokens = 3

                # compute a good length
                length = context_length - num_special_tokens

                src_seq = np.random.randint(
                    low=num_special_tokens + 1,  # skip pad
                    high=vocab_size - num_special_tokens - 1,
                    size=(batch_size, length),
                )
                tgt_seq = src_seq + 1  # add one to predict next

                # pad to total sequence
                padding = [pad] * (context_length - (1 + length + 1))
                x = np.concatenate([bos, src_seq, eos, *padding], axis=1)
                y = np.concatenate([bos, tgt_seq, eos, *padding], axis=1)

                yield x, y  # [batch_size, context_length], [batch_size, context_length]

        context_length = self.config.context_length
        example_sequence_shape = tf.TensorShape((batch_size, context_length))

        dataset = tf.data.Dataset.from_generator(
            _generate,
            output_types=(tf.int64, tf.int64),
            output_shapes=(example_sequence_shape, example_sequence_shape),
        )
        return dataset


class LanguageModelInputConfig:
    batch_size: int
    prefetch: int
    pack: bool = True
    split: str = "TRAIN"


class LanguageModelInput:
    """Wrapper class that acts as the input_fn to TPUEstimator."""

    def __init__(self, file_pattern: List[str]):
        logging.info("init ToyModelInput()")
        self._file_pattern = file_pattern

    def __call__(self, params):
        """Input function which provides a single batch for train or eval."""
        # Retrieves the batch size for the current shard. The # of shards is
        # computed according to the input pipeline deployment. See
        # `tf.estimator.tpu.RunConfig` for details.
        batch_size = params["batch_size"]
        max_seq_length = params["max_sequence_length"]

        logging.info(
            "call LanguageModelInput() with batch size {} and sequence length",
            batch_size,
            max_seq_length,
        )

        filenames = tf.io.gfile.glob(self._file_pattern)
        logging.info(
            "Found %s files matching %s" % (len(filenames), self._file_pattern)
        )
        if not filenames:
            raise ValueError("No matching files found")
        dataset = tf.data.TFRecordDataset(filenames, buffer_size=64 * 1024 * 1024)
        keys = ["target"]
        EOS = 1
        PAD = 0

        def decode_example(serialized_example):
            """Return a dict of Tensors from a serialized tensorflow.Example."""
            decoded = tf.io.parse_example(
                serialized=[serialized_example],
                features={k: tf.VarLenFeature(tf.int64) for k in keys},
            )
            decoded = {k: v.values for k, v in decoded.items()}
            # append EOS
            decoded = {k: tf.concat([v, [EOS]], 0) for k, v in decoded.items()}
            return decoded

        ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # pack the dataset
        ds = transformer.datasets.pack_or_pad(
            ds,
            sequence_length=max_seq_length,
            pack=params["pack"],
            feature_keys=None,
            ensure_eos=False,
        )
        # ds is
        if params["split"] == "TRAIN":
            if params["repeat"]:
                ds = ds.repeat()
            ds = ds.shuffle(1000)
            ds = ds.batch(batch_size, drop_remainder=True)
        else:
            ds = ds.batch(batch_size)
        if params["prefetch"]:
            ds = ds.prefetch(params["prefetch"])
        return ds
