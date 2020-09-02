import copy
import json
import os
from typing import Dict

import tensorflow as tf
from absl import app, logging
from absl.flags import argparse_flags
from pydantic.dataclasses import dataclass

import lm.config
import lm.devices
import lm.infeeds
import lm.models
from lm.devices.tpu import TPUInfeedSpec, TPUJobSpec

"Eval a saved model using a dataset"


@dataclass
class ScheduleSpec:
    steps: int


@dataclass
class EvalConfig:
    infeed: Dict
    model: Dict
    model_path: str
    schedule: ScheduleSpec
    device: lm.devices.DeviceSpec = lm.devices.CPUDeviceSpec()


class Evaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model = None
        self.infeed = None
        self.device = None

    def load_model(self):
        if not (self.model is None):
            return self.model
        self.model = lm.models.from_config(self.config.model)
        return self.model

    def load_infeed(self):
        if not (self.infeed is None):
            return self.infeed
        self.infeed = lm.infeeds.from_config(self.config.infeed)
        return self.infeed

    def create_jobspec(self):
        model = self.load_model()
        infeed = self.load_infeed()
        return TPUJobSpec(
            function=model,
            params={
                # patch legacy config
                "eval_steps": self.config.schedule.steps,
                "model_path": self.config.model_path,
                "steps_per_iteration": self.config.schedule.steps,
                "steps_per_checkpoint": self.config.schedule.steps,
                "max_sequence_length": infeed.dataset.context_length,
                "vocab_size": infeed.dataset.vocab_size,
            },
            max_steps=self.config.schedule.steps,
            use_tpu=type(self.config.device) is lm.devices.TPUDeviceSpec,
            model_path=self.config.model_path,
            # steps_per_iteration=self.config.schedule.steps_per_iteration,
            # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
            infeed=TPUInfeedSpec(
                batch_size=infeed.config.batch_size, function=infeed.eval, params={}
            ),
        )

    def execute(self, jobspec):
        if self.device is None:
            self.device = lm.devices.from_config(self.config.device)
        return self.device.execute(jobspec)


def parse_args(args, parser=None):
    # Parse command line arguments
    parser.add_argument(
        "runspec",
        type=str,
        help="the json file specifiing the configuration for this run",
    )  # Name of TPU to train on, if any
    parser.add_argument("--testrun", action="store_true", default=False)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="the location of the dataset to evaluate",
    )
    parser.add_argument("--check-dataset", action="store_true", default=False)
    parser.add_argument("--output", type=str, help="save results to a file")
    parser.add_argument("--tpu", type=str, help="Name of TPU to train on, (if any)")


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def convert_train_spec_to_eval_spec(settings):
    s = copy.deepcopy(settings)
    s.pop("other")
    s.pop("regularization")
    s.pop("runspec")
    s["schedule"].pop("steps_per_checkpoint")
    s["schedule"].pop("steps_per_iteration")
    return s


def main(args):
    logging.info("started evaluation process")

    settings = lm.config.load(args.runspec)

    if args.dataset:
        dscfg = lm.config.load(args.dataset)
        ds_location = os.path.split(args.dataset)[0] + "/*.tfrecord"
        settings["infeed"]["dataset"] = dscfg
        settings["infeed"]["file_pattern"] = ds_location
        settings["infeed"]["max_sequence_length"] = dscfg["context_length"]

    settings = convert_train_spec_to_eval_spec(settings)

    econfig = EvalConfig(**settings)

    # patch config
    if args.tpu:
        econfig.device = lm.devices.TPUDeviceSpec(address=args.tpu)

    evaluator = Evaluator(econfig)

    # if args.check_dataset:
    #     check_dataset(trainer, args)

    # saves config to logdir for experiment management
    # save_config(pprint.pformat(params), params["model_path"])
    # save_config(params, params["model_path"])

    j = evaluator.create_jobspec()
    j.eval = True
    results = evaluator.execute(j)

    if args.output:
        with tf.io.gfile.GFile(args.output, "w") as fd:
            jnice = {k: v.item() for k, v in results.items()}
            json.dump(jnice, fd, indent=2)

    logging.info("completed evaluation process")


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
