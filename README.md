---
title: Language Model (lm) End to End Pipeline 
---

[![image](https://img.shields.io/pypi/v/lm.svg)](https://pypi.python.org/pypi/lm)

[![image](https://img.shields.io/travis/NeuroArchitect/lm.svg)](https://travis-ci.com/NeuroArchitect/lm)

[![Documentation Status](https://readthedocs.org/projects/lm/badge/?version=latest)](https://lm.readthedocs.io/en/latest/?badge=latest)

[![Updates](https://pyup.io/repos/github/NeuroArchitect/lm/shield.svg)](https://pyup.io/repos/github/NeuroArchitect/lm/)

End to End Language Model Pipeline built for training speed

There are few frameworks out there that focus on sequence to sequence neural network models.
Most notables are the ones built by [Google](github.com/tensorflow/seq2seq) and [Facebook](github.com/pytorch/fairseq).
This repository focuses on seq2seq and language model (next token prediction) using an opinionated end to end setup.
The project objective is to create a *production* pipeline that runs end to end and contains all the professional steps required to achieve state of the art language models.

It leverages:
- mesh tensorflow to train on 8, 32, 256, 512 TPUs
- jsonnet configuration files
- docker/kubeflow for orchestrating the various experiments
- absl for process management, flags, unittest

It uses and supports *ONLY*: 
- Tensorflow (1.15)
- Tensorflow Mesh 
- TPUs (maybe GPUs cluster in the future, maybe)
- Docker / Kubeflow setup

# Sponsor
- contact to sponsor this project

# License
-   Free software: Apache Software License 2.0

# Credits
This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
