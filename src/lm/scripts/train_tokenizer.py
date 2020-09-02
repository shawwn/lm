"""
Trains a tokenizer over a rawtxt corpus
"""

import os
import random
import sys
from glob import glob

import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags

from lm.datasets import constants
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
)
from tokenizers.normalizers import NFKC, Sequence  # , Lowercase

# from tokenizers.pre_tokenizers import ByteLevel


def parse_flags(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Location of the dataset files. Files ending in .zst are treated as \
                        archives, all others as raw text. Can be a glob (/dataset/*.xz, /dataset/*.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Location to write the generated tokenizer configuration",
    )
    parser.add_argument(
        "--extra_tokens",
        type=str,
        help="location of a txt file with one word per line",
        required=False,
    )
    parser.add_argument(
        "--vocab_size", type=int, help="Size of vocabulary", required=True
    )
    parser.add_argument("--random_seed", type=int, default=1337, help="seed")
    args = parser.parse_args(argv[1:])
    return args


def listfiles(location):
    txt_files = list(p for p in glob(location) if not os.path.isdir(p))

    # try with general glob
    if not txt_files:
        txt_files = list(glob(os.path.join(location, "*.*")))

    txt_files = list(p for p in txt_files if not os.path.isdir(p))
    return txt_files


def setup_tokenizer(_):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    normalizers = [NFKC()]
    tokenizer.normalizer = Sequence(normalizers)
    return tokenizer


def main(args):

    random.seed(args.random_seed)

    txt_files = listfiles(args.input)
    if not txt_files:
        logging.error("no data files found")
        return

    os.makedirs(args.output, exist_ok=True)

    # setup
    tokenizer = setup_tokenizer(args)

    if args.extra_tokens:
        with tf.io.gfile.GFile(args.extra_tokens) as fd:
            words = [l.strip() for l in fd.readlines()]
        tokenizer.add_tokens(words)
        if args.vocab_size < len(words):
            logging.error("vocab size is less than the provided tokens. aborting")
            sys.exit(-1)

    # train
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[constants.PAD, constants.EOS],
    )
    tokenizer.train(trainer, txt_files)

    # save
    tokenizer_path = os.path.join(args.output, "byte-level-bpe.tokenizer.json")
    tokenizer.save(tokenizer_path, pretty=True)
    encoded_gold = tokenizer.encode("I can feel the magic, can you?")
    logging.info("tokenizer saved at %s", tokenizer_path)

    # test
    tokenizer = Tokenizer.from_file(tokenizer_path)
    encoded = tokenizer.encode("I can feel the magic, can you?")

    if not all(a == b for a, b in zip(encoded.ids, encoded_gold.ids)):
        logging.error("saved tokenizer and trained tokenizer do not match")

    tokenizer.model.save(args.output)

    logging.info("tokenizer model saved at %s", args.output)


def apprun():
    app.run(main, flags_parser=parse_flags)


if __name__ == "__main__":
    apprun()
