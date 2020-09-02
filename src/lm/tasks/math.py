import numpy as np
import tensorflow as tf
from pydantic.dataclasses import dataclass
from pydantic import BaseModel

from .base import BaseTask, BaseTaskConfig, BaseTaskDatasetConfig

import lm

class AdditionDatasetConfig(BaseTaskDatasetConfig):
    kind:str
    context_length: int 
    seed: int
    ndigits: int = 3
    vocab_size: int = 10

class AdditionConfig(BaseTaskConfig):
    name: str
    description: str
    dataset: AdditionDatasetConfig

@lm.register_task('lm.tasks.Addition', AdditionConfig)
class Addition(BaseTask):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(AdditionConfig(**kwds)))

    def gen_serialization(self, ndigit):
        def serialize(tokens, idx):
            """
            Creates a tf.Example message ready to be written to a file.
            """

            def _int64_list_feature(int_list):
                """Returns an int64_list from a bool / enum / int / uint."""
                return tf.train.Feature(int64_list=tf.train.Int64List(value=int_list))

            # Create a dictionary mapping the feature name to the tf.Example-compatible
            # data type.
            feature = {
                "tokens": _int64_list_feature(tokens),
                "idx": _int64_list_feature(idx),
            }

            example = tf.train.Example(features=tf.train.Features(feature=feature))

            return example.SerializeToString()

        feature_spec = {
            "tokens": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
            "idx": tf.io.FixedLenFeature([ndigit * 3], dtype=tf.dtypes.int64),
        }

        def deserialize(example):
            return tf.io.parse_single_example(example, features=feature_spec)

        return serialize, deserialize

    def gen_dataset(self, ndigit):
        ndigit = self.config.ndigit
        # vocab_size = 10 # 10 possible digits 0..9
        # # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        # block_size = ndigit + ndigit + ndigit + 1 - 1

        # split up all addition problems into either training data or test data
        num = (10 ** ndigit) ** 2  # total number of possible combinations
        r = np.random.RandomState(1337)  # make deterministic
        perm = r.permutation(num)
        num_test = min(
            int(num * 0.2), 1000
        )  # 20% of the whole dataset, or only up to 1000
        test = perm[:num_test]
        train = perm[num_test:]
        return train, test

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        ndigit = self.ndigit
        nd = 10 ** ndigit
        a = idx // nd
        b = idx % nd
        c = a + b
        render = f"%0{ndigit}d%0{ndigit}d%0{ndigit+1}d" % (
            a,
            b,
            c,
        )  # e.g. 03+25=28 becomes "0325028"
        dix = [int(s) for s in render]  # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = np.array(dix[:-1], dtype=np.int32)
        y = np.array(dix[1:], dtype=np.int32)  # predict the next token in the sequence
        y[
            : ndigit * 2 - 1
        ] = (
            -100
        )  # we will only train in the output locations. -100 will mask loss to zero
        return (x, y, render)

    def humanize(self, value: str):
        ndigit = self.config.ndigit
        a = value[:ndigit]
        b = value[ndigit : ndigit * 2]
        c = value[ndigit * 2 :]
        return "%d + %d = %d" % (int(a), int(b), int(c))

    def __call__(self, ndigit=3):
        ndigit = 3
        all_sum_train, all_sums_test = self.gen_dataset(ndigit)
        for example in all_sums_test:
            tokens, indices, render = self.gen_example(example, ndigit=ndigit)
            expression = self.humanize(render)
            print(expression, "is: ", eval(expression.replace("=", "==")))
            break

    def example_generator(self):
        with datasets.tfrecord.generator():
            pass


class SumOneDatasetConfig(BaseTaskDatasetConfig):
    kind:str
    context_length: int 
    seed: int
    vocab_size: int = 10

class SumOneConfig(BaseTaskConfig):
    dataset: SumOneDatasetConfig

@lm.register_task('lm.tasks.SumOne', SumOneConfig)
class SumOne(BaseTask):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(SumOneConfig(**kwds)))
    
    def infeed(self):
        ds = lm.get_dataset(self.dataset)