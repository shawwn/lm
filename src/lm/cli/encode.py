import os
import time
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from tqdm import auto as tqdm

import lm.config
import lm.encoders
import lm.examples

args = None

def readlines_txt(src):
    with open(src) as fd:
        if not args.by_line:
            return [fd.read()]
        else:
            return fd.readlines()


LINE_READER = {
    ".txt": readlines_txt,
    ".tsv": readlines_txt,
}


def readlines(src):
    _, ext = os.path.splitext(src)
    f = LINE_READER.get(ext, None)
    if f is None:
        logging.warning("no readlines for file %s", src)
        return
    return f(src)


# Helper functions and classes
def sizechunks(l, n):
    out = []
    chunk = []
    sz = 0
    for fpath in tqdm.tqdm(l, label="Measuring size..."):
        chunk.append(fpath)
        sz += tf.io.gfile.stat(fpath).length
        if sz >= n:
            out.append(chunk)
            sz = 0
            chunk = []
    if chunk:
        out.append(chunk)
    return out


def parallel(src_dst_list, total):
    count = args.nproc or cpu_count()
    pool = Pool(processes=count) if count > 1 else None
    mapper = pool.imap if count > 1 else map
    if args.format == "tfrecord":
      transformer = lm.examples.transform_many_and_write_one_tfrecord
    elif args.format in ["tok16", "tok32"]:
      transformer = lm.examples.transform_many_and_write_one_tok16_or_tok32
    else:
      raise ValueError("Unknown --format {}".format(args.format))
    token_total = 0
    example_total = 0
    for token_count, example_count in tqdm.tqdm(
        mapper(transformer, src_dst_list),
        total=total,
    ):
        token_total += token_count
        example_total += example_count
    return token_total, example_total


def parse_args(args, parser):
    parser.add_argument(
        "input",
        type=str,
        help="A file containing a list of filenames. Each file will become a single training example (unless --by_line is set).",
    )
    parser.add_argument(
        "output", type=str, default="output", help="Where to write tfrecords"
    )

    parser.add_argument(
        "--size",
        type=float,
        default=165.0,
        help="the size in MB of uncompressed text to add to each tfrecord file, default 165MB",
    )
    parser.add_argument(
        "--name", type=str, default="dataset", help="prefix name for the output files."
    )
    parser.add_argument(
        "--encoder", type=str, default="gpt2", help="Name or path of an encoder spec"
    )
    parser.add_argument(
        "--by_line", action="store_true", help="encodes each line as a separate record"
    )
    parser.add_argument(
        "--no-ftfy", action="store_true", help="Don't pass source text through ftfy.fix_text() (and don't replace unicode ellipses with '...')"
    )
    parser.add_argument(
        "--nproc", type=int, default=0, help="the number of processes to use for multiprocess encoding (0=all CPUs, 1=disable multiprocessing)"
    )
    parser.add_argument(
        "--format", type=str, default="tfrecord", help="""--format=tfrecord (the default) writes tokens as .tfrecord files; each document becomes a single tf.train.Example. --format=tok16 simply dumps uint16 tokens to disk; good for OpenAI GPT-2/GPT-3 tokenization (which is the default --encoder setting). --format=tok32 dumps int32 tokens to disk; suitable for advanced tokenization schemes that need to store negative-valued tokens or whose vocabulary is larger than 65535 elements."""
    )


def is_integer(x):
  return np.can_cast(x, np.int32)


def is_float(x):
  return np.can_cast(x, np.float32)


def is_exact(x):
  return is_integer(x) or is_float(x) and x == int(x)


def num(x, digits_after_decimal=2):
  if is_integer(x):
    spec = '{:,d}'
  else:
    spec = '{:,.%df}' % digits_after_decimal
  return spec.format(x)


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(argv):
    global args
    args = argv

    txt_files = open(args.input).read().splitlines()
    if not txt_files:
        logging.error("no data files found")
        return

    os.makedirs(args.output, exist_ok=True)

    if tf.io.gfile.exists(args.encoder):
        enccfg = lm.config.load(args.encoder)
        encoder = lm.encoders.from_config(enccfg)
    else:
        encoder = lm.encoders.from_config(dict(kind="hf", location=args.encoder))

    megabytes_per_tfrecord = int(args.size * 1e6)
    file_chunks = sizechunks(
        txt_files, megabytes_per_tfrecord
    )  # Assign files_per file to a tfrecord file each

    logging.info(
        "Got %d files, divided into %d chunks.", len(txt_files), len(file_chunks)
    )

    def getdst(name, idx, total):
        filename_format = '%s-%05d-of-%05d' # standard tensorflow shard naming convention: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sharded-filename
        if args.format == "tfrecord":
          return os.path.join(args.output, (filename_format + ".tfrecord") % (name, idx, total))
        elif args.format == "tok16":
          return os.path.join(args.output, (filename_format + ".tok16") % (name, idx, total))
        elif args.format == "tok32":
          return os.path.join(args.output, (filename_format + ".tok32") % (name, idx, total))
        else:
          raise ValueError("Unknown --format {}".format(args.format))

    jobs = list(
        (encoder, chunks, getdst(args.name, idx, len(file_chunks)), args)
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    token_total, example_total = parallel(jobs, total=len(file_chunks))
    end = time.time()
    elapsed = (end - start)
    tokens_per_second = token_total / elapsed
    tokens_per_file = token_total / len(jobs)

    logging.info(
        "finished in %ss: tokenized %d of %d files (%s tokens @ %s tokens/sec) into %d files (~%s tokens per file)",
        num(elapsed), example_total, len(txt_files), num(token_total), num(tokens_per_second), len(jobs), num(tokens_per_file),
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
