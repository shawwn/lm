import argparse
import collections
import importlib
import json
import os
import random
import sys
import numpy as np

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from absl import app, logging
from absl.flags import argparse_flags

import mesh_tensorflow as mtf
import tensorflow as tf

from tensorflow.compat import v1

"""Main LM command line"""

SUBCOMMANDS = {}


def register_subcommand(module_name):
    m = importlib.import_module("lm.cli." + module_name)
    SUBCOMMANDS[module_name] = m

def set_random_seed(args):
    seed = args.seed
    random.seed(args.seed)
    v1.set_random_seed(args.seed)
    np.random.seed(args.seed)

def fix_logger():
    l = v1.get_logger()
    l.propagate = False

def parse_args(args):
    # Parse command line arguments
    parser = argparse_flags.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337, help="seed")

    subparsers = parser.add_subparsers(help="Available commands", dest="subparser")
    for name, cmd in SUBCOMMANDS.items():
        cmd_parser = subparsers.add_parser(
            name,
            help=cmd.__doc__,
            # Do not make absl FLAGS available after the subcommand `roll_dice`.
            inherited_absl_flags=False,
        )
        cmd.parse_args(args, cmd_parser)

    args = parser.parse_args(args[1:])
    set_random_seed(args)
    fix_logger()
    return args



def main(args):
    cmd = SUBCOMMANDS.get(args.subparser, None)
    if cmd is None:
        app.usage(shorthelp=True, exitcode=-1)
        return
        # raise ValueError('invalid command %s', args.subparser)
    return cmd.main(args)


def apprun():
    tf.disable_v2_behavior()

    register_subcommand("preprocess")
    register_subcommand("cleantxt")
    register_subcommand("configure")
    register_subcommand("train")
    register_subcommand("eval")
    register_subcommand("interactive")
    register_subcommand("synth")
    app.run(main, flags_parser=parse_args)


if __name__ == "__main__":
    app()
