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


def readlines_txt(src):
    with open(src) as fd:
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
    count = cpu_count() - 1 or 1
    pool = Pool(processes=count)
    ret = 0
    for i in tqdm.tqdm(
        pool.imap(lm.examples.transform_many_and_write_one_tfrecord, src_dst_list),
        total=total,
    ):
        ret += i
    return ret


def listfiles(location):
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
        help="Path to where your files are located. Files ending in .zst are treated as \
                        archives, all others as raw text.",
    )
    parser.add_argument(
        "output", type=str, default="output", help="Where to write tfrecords"
    )

    parser.add_argument(
        "--size",
        type=int,
        default=50 * 2 ** 20,
        help="the size in MiB of uncompressed text to add to each tfrecord file, default 50MiB",
    )
    parser.add_argument(
        "--name", type=str, default="dataset", help="prefix name for the output files."
    )
    parser.add_argument(
        "--encoder", type=str, required=True, help="Name or path of an encoder spec"
    )


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(args):

    txt_files = listfiles(args.input)
    if not txt_files:
        logging.error("no data files found")
        return

    os.makedirs(args.output, exist_ok=True)

    if tf.io.gfile.exists(args.encoder):
        enccfg = lm.config.load(args.encoder)
        encoder = lm.encoders.from_config(enccfg)
    else:
        encoder = lm.encoders.from_config(dict(kind="hf", location=args.encoder))

    file_chunks = sizechunks(
        txt_files, args.size
    )  # Assign files_per file to a tfrecord file each

    logging.info(
        "Got %d files, divided into %d chunks.", len(txt_files), len(file_chunks)
    )

    def getdst(name, idx, total):
        return os.path.join(args.output, "%s_%05d_%05d.tfrecord" % (name, idx, total))

    jobs = (
        (encoder, chunks, getdst(args.name, idx, len(file_chunks)))
        for idx, chunks in enumerate(file_chunks)
    )

    start = time.time()
    ret = parallel(jobs, total=len(txt_files))
    end = time.time()

    logging.info(
        "job completed in %.2fs, %d / %d good files.", end - start, ret, len(txt_files)
    )


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
