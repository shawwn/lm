from typing import Dict
import functools
import collections

REGISTRY = collections.defaultdict(list)


def register_model(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["models"][model_name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register_datasets(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["datasets"][model_name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


def register_infeeds(f, name, config):
    assert not (name in REGISTRY), "model with that name already present"
    REGISTRY["infeeds"][model_name] = f

    @functools.wraps(f)
    def wrapper(*args, **kwds):
        return f(*args, **kwds)

    return wrapper


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
