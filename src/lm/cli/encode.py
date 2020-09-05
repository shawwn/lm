import os
import time
from glob import glob
from multiprocessing import Pool, cpu_count

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
    for fpath in l:
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
    count = args.nproc
    pool = Pool(processes=count) if count > 1 else None
    mapper = pool.imap if count > 1 else map
    token_total = 0
    example_total = 0
    for token_count, example_count in tqdm.tqdm(
        mapper(lm.examples.transform_many_and_write_one_tfrecord, src_dst_list),
        total=total,
    ):
        token_total += token_count
        example_total += example_count
    return token_total, example_total


def listfiles(location):
    # check if is an index file
    txt_files = []
    if tf.io.gfile.exists(location):
        if not tf.io.gfile.isdir(location):
            with tf.io.gfile.GFile(location) as fd:
                for l in fd.readlines():
                    if tf.io.gfile.exists(l):
                        txt_files.append(l)
    if txt_files:
        return txt_files

    txt_files = list(p for p in glob(location) if not os.path.isdir(p))

    # try with general glob
    if not txt_files:
        txt_files = list(glob(os.path.join(location, "*.*")))

    txt_files = list(p for p in txt_files if not os.path.isdir(p))
    return txt_files


def parse_args(args, parser):
    parser.add_argument(
        "input",
        type=str,
        help="Path to where your files are located.",
    )
    parser.add_argument(
        "output", type=str, default="output", help="Where to write tfrecords"
    )

    parser.add_argument(
        "--size",
        type=float,
        default=50.0,
        help="the size in MB of uncompressed text to add to each tfrecord file, default 50MiB",
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
        "--nproc", type=int, default=cpu_count(), help="the number of processes to use for multiprocess encoding (<= 1 to disable multiprocessing)"
    )


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
        return os.path.join(args.output, "%s_%05d_%05d.tfrecord" % (name, idx, total))

    jobs = (
        (encoder, chunks, getdst(args.name, idx, len(file_chunks)), args)
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    token_total, example_total = parallel(jobs, total=len(file_chunks))
    end = time.time()

    logging.info(
        "job completed in %.2fs, %d / %d good files, %d tokens.", end - start, example_total, len(txt_files), token_total
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
