"Module that describes simple addition tasks"
import numpy as np

import lm

from .base import BaseTask, BaseTaskConfig, BaseTaskDatasetConfig


class AdditionDatasetConfig(BaseTaskDatasetConfig):
    context_length: int
    seed: int
    ndigits: int = 3
    vocab_size: int = 10


class AdditionConfig(BaseTaskConfig):
    dataset: AdditionDatasetConfig


@lm.register_task("lm.tasks.Addition", AdditionConfig)
class Addition(BaseTask):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(AdditionConfig(**kwds)))

    def build_infeed(self):
        if self.infeed.kind.endswith("Generator"):
            dsgen = lm.get_infeed(self.infeed)
            dsgen.set_producer(AdditionProducer(**self.dataset))
            return dsgen

    def humanize(self, value: str):
        ndigit = self.config.ndigit
        a = value[:ndigit]
        b = value[ndigit : ndigit * 2]
        c = value[ndigit * 2 :]
        return "%d + %d = %d" % (int(a), int(b), int(c))


@lm.register_dataset("lm.datasets.AdditionProducer", None)
class AdditionProducer:
    """
    Encodes a simple equation A + B = C in an encoder friendly format
    inspired by github.com/karpathy/mingGPT
    """

    def __init__(self, **kwds):
        super().__init__()
        self.config = AdditionDatasetConfig(**kwds)

    def gen_dataset(self, ndigit):
        ndigit = self.config.ndigits
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

    # def __call__(self, ndigit=3):
    #     ndigit = 3
    #     all_sum_train, all_sums_test = self.gen_dataset(ndigit)
    #     for example in all_sums_test:
    #         tokens, indices, render = self.gen_example(example, ndigit=ndigit)
    #         expression = self.humanize(render)
    #         print(expression, "is: ", eval(expression.replace("=", "==")))
    #         break


class Seq2SeqDataset(BaseTaskDatasetConfig):
    max_sequence_length: int
    vocab_size: int = 10


class SumOneDatasetConfig(Seq2SeqDataset):
    seed: int


class SumOneConfig(BaseTaskConfig):
    dataset: SumOneDatasetConfig


@lm.register_dataset("lm.datasets.SumOneGen", SumOneDatasetConfig)
class SumOneGen:
    def __init__(self, **kwds):
        super().__init__()
        self.config = SumOneDatasetConfig(**kwds)

    def __call__(self):
        vocab_size = self.config.vocab_size
        context_length = self.config.max_sequence_length
        np.random.seed(self.config.seed)

        def generate_single_example():
            while True:
                # special tokens
                shape = (1,)
                pad = np.full(shape, 0)  # pad token
                eos = np.full(shape, 1)  # end of sentence token
                bos = np.full(shape, 2)  # begin of sentence token
                num_special_tokens = 3

                # compute a good length
                length = context_length - num_special_tokens

                src_seq = np.random.randint(
                    low=num_special_tokens + 1,  # skip pad
                    high=vocab_size - num_special_tokens - 1,
                    size=(length,),
                )
                tgt_seq = src_seq + 1  # add one to predict next

                # pad to total sequence
                padding = [pad] * (context_length - (1 + length + 1))
                x = np.concatenate([bos, src_seq, eos, *padding], axis=0)
                y = np.concatenate([bos, tgt_seq, eos, *padding], axis=0)

                yield lm.examples.Seq2SeqSimpleExample(x, y).serialize()

        return generate_single_example


@lm.register_task("lm.tasks.SumOne", SumOneConfig)
class SumOne(BaseTask):
    def __init__(self, **kwds):
        super().__init__()
        self.__dict__.update(dict(SumOneConfig(**kwds)))

    def build_generator(self):
        return SumOneGen(**self.dataset.dict())

    def build_infeed(self):
        # base on the kind of infeed we create the right dataset
        if self.infeed.kind.endswith("Generator"):
            dsgen = lm.get_infeed(self.infeed)
            dsgen.set_producer(SumOneGen(**self.dataset))
            return dsgen
        # generator = self.build_gen_func()
