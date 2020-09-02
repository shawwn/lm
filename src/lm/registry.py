import collections
import functools
from typing import Dict
from absl import logging

REGISTRY = collections.defaultdict(dict)
CLASS_NAMES = set()

def register_model(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["models"][name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register_dataset(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["datasets"][name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register_infeed(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["infeeds"][name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register_encoder(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["encoders"][name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register(cls, kind, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY[kind][name] = cls

    @functools.wraps(cls)
    def wrapper(*args, **kwds):
        return cls(*args, **kwds)

    return wrapper

def register_task(name, config=None):
    logging.info('task %s registered', name)
    return functools.partial(register, kind='tasks', name=name, config=config)

def model_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["models"][model](**config)


def infeed_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["infeeds"][model](**config)


def dataset_from_config(config: Dict):
    model = config["kind"]
    return REGISTRY["datasets"][model](**config)


# def from_config(config: Dict):
#     cfg = InfeedConfig(**config)
#     infeed = Seq2SeqTFRecordInfeed(cfg)
#     return infeed

def get_task(config):
    kind = config.get('kind', None)
    if kind is None:
        raise ValueError('invalid task configuration. "kind" key was not found in dictionary')
    return REGISTRY['tasks'][kind](**config)