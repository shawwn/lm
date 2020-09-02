import json
import os
import sys

import tensorflow as tf
from absl import logging

import _jsonnet

"Utility module to deal with resolving the system configuration"


def load(resource: str):
    if tf.io.gfile.isdir(resource):
        if tf.io.gfile.exists(os.path.join(resource, "dataset.info.json")):
            resource = os.path.join(resource, "dataset.info.json")
        if tf.io.gfile.exists(os.path.join(resource, "merges.txt")):
            # is a tokenizer directory
            return {
                "kind": "hf",
                "location": resource,
            }
        raise ValueError(
            "directory %s provided but directory does not contain any info" % resource
        )
    path, ext = os.path.splitext(resource)
    if ext in (".json",):
        with tf.io.gfile.GFile(resource) as fd:
            params = json.load(fd)
    elif ext in (".jsonnet",):
        try:
            json_str = _jsonnet.evaluate_file(
                resource,
                ext_vars={"MODEL_PATH": "Bob"},
                # import_callback=import_callback,
            )
            params = json.loads(json_str)
        except RuntimeError as e:
            logging.error(e)
            sys.exit(-1)
    else:
        raise ValueError
    return params
