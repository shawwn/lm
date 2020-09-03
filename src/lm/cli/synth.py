import json
import os

import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from pydantic.dataclasses import dataclass

import lm.config
import lm.infeeds.seq2seq
import lm.tasks
import lm.tf
from tqdm import auto as tqdm

def parse_args(_, parser):
    # Parse command line arguments
    parser.add_argument(
        "taskspec",
        type=str,
        help="the json file specifiing the configuration for this run",
    )  # Name of TPU to train on, if any
    parser.add_argument(
        "output", type=str, help="processes the dataset and saves is to this location"
    )
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument(
        "--max_seq_len",
        type=int,
        help="Also called context size. The max input sequence of the neural network.",
    )


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(args):
    logging.info("started synth process")

    task_dict = lm.config.load(args.taskspec)
    logging.info(task_dict)
    task = lm.registry.get_task(task_dict)
    if args.vocab_size:
        task.dataset.vocab_size = args.vocab_size
    if args.max_seq_len:
        task.dataset.max_sequence_length = args.max_seq_len

    # save info
    tf.io.gfile.makedirs(args.output)
    output_location = os.path.join(args.output, "dataset.info.json")
    with tf.io.gfile.GFile(output_location, "w") as w:
        json.dump(task.dataset.dict(), w, indent=2)
        logging.info("wrote %s", output_location)

    # generate data
    gen_fn = task.build_generator()

    infeed = lambda params: lm.tf.from_generator(gen_fn)
    params = {}
    output_location = os.path.join(args.output, "synth_%05d.tfrecord" % 1)
    with tf.io.TFRecordWriter(output_location) as w:
        it = iter(lm.tf.consume(infeed, params))
        for _ in tqdm.tqdm(range(args.n_samples)):
            ex = next(it)
            w.write(ex)

    # train
    logging.info("completed synth process. dataset generated %s", args.output)


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
