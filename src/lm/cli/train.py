import datetime
import json
import os
from typing import Dict

import tensorflow as tf
import tensorflow.compat.v1 as v1
from absl import app, logging
from absl.flags import argparse_flags
from pydantic.dataclasses import dataclass

import lm.config
import lm.datasets
import lm.devices
import lm.infeeds
import lm.models
from lm.devices.tpu import TPUInfeedSpec, TPUJobSpec
from lm.tasks.base import BaseTaskConfig


def serving_input_receiver_fn():
    feature = tf.placeholder(tf.int32, shape=[None, 8], name="tokens")
    return tf.estimator.export.TensorServingInputReceiver(feature, feature)


@dataclass
class ScheduleSpec:
    steps: int
    steps_per_checkpoint: int
    steps_per_iteration: int


@dataclass
class GradientSpec:
    learning_rate: Dict
    optimizer: Dict
    gradient_clipping: float
    weight_decay: float


@dataclass
class TrainerConfig:
    device: Dict
    infeed: Dict
    model: Dict
    # other: Dict
    # regularization: Dict
    gradients: GradientSpec
    schedule: ScheduleSpec
    # checkpoints_location: str
    model_path: str
    task: Dict


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        # self.device = "tpu" if config.tpu else "cpu"
        self.infeed = None
        self.model = None
        self.device = None

    def save_checkpoint(self):
        state = self.model.state_dict()
        logging.info("saving model checkpoint to %s", self.config.ckpt_path)
        self.save(state, self.config.ckpt_path)
        logging.info("saved model checkpoint to %s", self.config.ckpt_path)

    def load_model(self):
        if not (self.model is None):
            return self.model
        self.model = lm.get_model(self.config.model)
        return self.model

    def load_infeed(self):
        if not (self.infeed is None):
            return self.infeed
        self.infeed = lm.infeeds.get_infeed(self.config.infeed)
        return self.infeed

    def create_train_jobspec(self):
        model = self.load_model()
        infeed = self.load_infeed()
        return TPUJobSpec(
            function=model,
            params={
                # patch legacy config
                "opt_name": self.config.runspec.optimizer["name"],
                "train_steps": self.config.schedule.steps,
                "steps_per_checkpoint": self.config.schedule.steps_per_checkpoint,
                "steps_per_iteration": self.config.schedule.steps_per_iteration,
                "model_path": self.config.model_path,
                "vocab_size": infeed.dataset.vocab_size,
                "max_sequence_length": infeed.config.max_sequence_length,
                **self.config.runspec.optimizer,
                **self.config.runspec.learning_rate,
            },
            max_steps=self.config.schedule.steps,
            use_tpu=self.config.device.get("kind", "cpu") == "tpu",
            model_path=self.config.model_path,
            # steps_per_iteration=self.config.schedule.steps_per_iteration,
            # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
            infeed=TPUInfeedSpec(
                batch_size=infeed.config.batch_size, function=infeed.train, params={}
            ),
            export="/tmp/export",
            signature=serving_input_receiver_fn,
        )

        def create_export_jobspec(self):
            model = self.load_model()
            infeed = self.load_infeed()
            EOS = 1

            return TPUJobSpec(
                function=model,
                params={
                    # patch legacy config
                    "opt_name": self.config.runspec.optimizer["name"],
                    "train_steps": self.config.schedule.steps,
                    "steps_per_checkpoint": self.config.schedule.steps_per_checkpoint,
                    "steps_per_iteration": self.config.schedule.steps_per_iteration,
                    "model_path": self.config.model_path,
                    "stop_at_token": EOS,
                    **self.config.runspec.optimizer,
                    **self.config.runspec.learning_rate,
                },
                max_steps=self.config.schedule.steps,
                use_tpu=self.config.device.get("kind", "cpu") == "tpu",
                model_path=self.config.model_path,
                # steps_per_iteration=self.config.schedule.steps_per_iteration,
                # steps_per_checkpoint=self.config.schedule.steps_per_checkpoint,
                infeed=TPUInfeedSpec(
                    batch_size=infeed.config.batch_size,
                    function=infeed.train,
                    params={},
                ),
                export="/tmp/export",
                signature=serving_input_receiver_fn,
            )

    def execute(self, jobspec):
        if self.device is None:
            self.device = lm.devices.from_config(self.config.device)
        return self.device.execute(jobspec)


def check_dataset(trainer, args):
    steps = trainer.config.schedule.steps
    infeed = trainer.load_infeed()

    logging.info("running for %d steps", steps)
    with v1.Session(graph=tf.Graph()) as sess:
        ds = infeed({"batch_size": infeed.config.batch_size})

        it = ds.make_one_shot_iterator()
        example = it.get_next()
        for i in range(steps):
            try:
                result = sess.run(example)
                logging.info("%d/%d: %r", i, steps, result)
            except tf.errors.OutOfRangeError:
                logging.error(
                    "dataset ended prematurely after only %d of the %d expected steps",
                    i,
                    steps,
                )


from contextlib import ContextDecorator, ExitStack

class TrainerCPUConfig:
    pass

class TrainerCPU(ContextDecorator):

    def __init__(self, config: TrainerCPUConfig):
        self.config = config
        self._sess = None
        self._graph = None

    def __enter__(self):
        print("creating graph")
        self._graph = tf.Graph()
        self._graph.as_default().__enter__()
        self._sess = tf.Session(graph=self._graph)
        self._sess.__enter__()

    def __exit__(self, exc_type, exc, exc_tb):
        self._sess = None

    def __call__(self, add_example_producer):
        with ExitStack(), self:
            ds = add_example_producer()
            inputs = self.add_input_pipeline(ds)
            model = self.add_model(inputs)
            for epoch in self.epoch_iterator():
                for result in self.step(model, inputs):
                    yield result

    def add_model(self, inputs):
        model_factory = model_factory_registry[self.config.model_name].from_config(
            self.config.model_config
        )
        model_graph = model_factory(inputs, params={})
        return model_graph

    def add_input_pipeline(self, dataset):
        it = tf.data.make_one_shot_iterator(ds)
        x = it.get_next()
        return x

    def step(self, model, example_batch):
        # for example_batch in dataset:
        while True:
            try:
                yield self._sess.run(inputs=example_batch, outputs=model.outputs)
            except tf.errors.OutOfRangeError:
                break

    def epoch_iterator(self):
        for e in range(self.config.experiment_config.num_epochs):
            yield


def parse_args(args, parser=None):
    # Parse command line arguments
    parser.add_argument(
        "trainspec",
        type=str,
        help="the json file specifiing the configuration for this run",
    )
    parser.add_argument(
        "--save-settings",
        type=str,
        help="freeze and save the final configuration settings.",
    )
    parser.add_argument("--check-dataset", action="store_true", default=False)
    parser.add_argument(
        "--device",
        type=str,
        help="Name of the device to train on, (TPUv3-8, v3-32, etc if any)",
    )
    parser.add_argument("--steps", type=int, help="max steps to run the train")
    parser.add_argument(
        "--dataset", type=str, help="location to a dataset jsonnet configuration file."
    )
    parser.add_argument(
        "--task", type=str, help="location to a task jsonnet configuration file."
    )
    # parser.add_argument(
    #     "--project",
    #     default=None,
    #     help="Project name for the Cloud TPU-enabled project. If not specified, we "
    #     "will attempt to automatically detect the GCE project from metadata.")

    # parser.add_argument(
    #     "--zone",
    #     default=None,
    #     help="GCE zone where the Cloud TPU is located in. If not specified, we "
    #     "will attempt to automatically detect the GCE project from metadata.")


def local_parse_args(args):
    parser = argparse_flags.ArgumentParser()
    parse_args(args, parser)
    return parser.parse_args(args[1:])


def main(args):
    logging.info("started train process")

    settings = lm.config.load(args.trainspec)

    # patch config
    if args.task:
        settings["task"] = lm.config.load(args.task)

    if args.dataset:
        dscfg = lm.config.load(args.dataset)
        ds_location = os.path.split(args.dataset)[0] + "/*.tfrecord"
        settings["infeed"]["dataset"] = dscfg
        settings["infeed"]["file_pattern"] = ds_location
        settings["infeed"]["max_sequence_length"] = dscfg["max_sequence_length"]

    if args.device:
        settings["device"]["kind"] = "tpu"
        settings["device"]["address"] = args.device

    if args.steps:
        settings["schedule"]["steps"] = args.steps

    logging.info("final config %r", settings)

    if args.save_settings:
        runspec = args.save_settings
    else:
        dt = datetime.datetime.now().strftime("%Y%M%d_%H%M%S")
        runspec = "run-%s.json" % dt

    with tf.io.gfile.GFile(runspec, "w") as fd:
        json.dump(settings, fd, indent=2)

    # reload the settings from the save configuration
    settings = lm.config.load(runspec)

    tconfig = TrainerConfig(**settings)
    trainer = Trainer(tconfig)

    if args.check_dataset:
        check_dataset(trainer, args)

    trainer.load_model()

    j = trainer.create_train_jobspec()
    j.train = True
    trainer.execute(j)

    # train
    logging.info("completed train process")


if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main, flags_parser=local_parse_args)
