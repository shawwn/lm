"Utility module to deal with resolving the system configuration"

import json
import os
import pkgutil
import sys

import tensorflow as tf
from absl import logging

import _jsonnet

LIBRARY_PATHS = []


def register_path(path):
    if path in LIBRARY_PATHS:
        return
    LIBRARY_PATHS.append(path)


#  Returns content if worked, None if file not found, or throws an exception
def try_path(location, rel):
    if not rel:
        raise RuntimeError("Got invalid filename (empty string).")
    if rel[0] == "/":
        full_path = rel
    else:
        full_path = location + rel
    if full_path[-1] == "/":
        raise RuntimeError("Attempted to import a directory")

    if not os.path.isfile(full_path):
        return full_path, None
    with open(full_path) as f:
        return full_path, f.read()


def import_callback(location, rel):
    # try relative first
    full_path, content = try_path(location, rel)
    if content:
        return full_path, content
    # standard library

    data = pkgutil.get_data(__name__, "jsonnet/" + rel)
    if data:
        return "__stdlib__", data.decode("utf-8")

    # registered lib path
    for libpath in LIBRARY_PATHS:
        full_path, content = try_path(libpath, rel)
        if content:
            return full_path, content
    raise RuntimeError("File not found")


def load(resource: str):
    if tf.io.gfile.isdir(resource):
        if tf.io.gfile.exists(os.path.join(resource, "dataset.info.json")):
            resource = os.path.join(resource, "dataset.info.json")
        elif tf.io.gfile.exists(os.path.join(resource, "merges.txt")):
            # is a tokenizer directory
            return {
                "kind": "hf",
                "location": resource,
            }
        else:
            raise ValueError(
                "directory %s provided but directory does not contain any info"
                % resource
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
                import_callback=import_callback,
            )
            params = json.loads(json_str)
        except RuntimeError as e:
            logging.error(e)
            sys.exit(-1)
    else:
        raise ValueError
    return params
