---
title: Language Model (lm) End to End Pipeline 
---

[![image](https://img.shields.io/pypi/v/lm.svg)](https://pypi.python.org/pypi/lm)

[![image](https://img.shields.io/travis/NeuroArchitect/lm.svg)](https://travis-ci.com/NeuroArchitect/lm)

[![Documentation Status](https://readthedocs.org/projects/lm/badge/?version=latest)](https://lm.readthedocs.io/en/latest/?badge=latest)

[![Updates](https://pyup.io/repos/github/NeuroArchitect/lm/shield.svg)](https://pyup.io/repos/github/NeuroArchitect/lm/)

# EleutherAI GPT Encoder

Turns files into OpenAI-tokenized .tfrecords, with one example per file.

## Encoder Quickstart

```
git clone https://github.com/shawwn/lm
cd lm
sudo pip3 install jsonnet==0.16.0
sudo pip3 install pydantic
sudo pip3 install transformers
sudo pip3 install pyfarmhash
bash tokenize.sh list_of_files.txt wherever_you_want_tfrecords/
```

Add e.g. `--size 500` to add 500MB of uncompressed input text into
each tfrecord file. (Note that it's `500 * 1e6` bytes, not `500 *
2**20` bytes.)

Set `--nproc 1` to disable multiprocessing. Useful for debugging via
pdb.

WARNING: The script currently DOES NOT check whether the output dir
already has some tfrecord files. It will overwrite existing files.
Meaning, if it generates fewer tfrecords than exist in the output dir,
you will end up with a weird partial dataset. I'll fix this soon, but
for now just be sure the output dir is empty. I probably could have
fixed this more quickly than I typed this.

---

The rest of the README is the unmodified version from the base `lm`
repo. Probably **disregard the rest of this README** if you're just
encoding text. But `lm` has some cool features if you're feeling
adventurous:

# TLDR

```
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e . 
lm cleantxt 
lm encode 
!mkdir -p /tmp/datasets/tfrecords/

ENCODE_INPUT=/tmp/datasets/txt
ENCODE_OUTPUT= /tmp/datasets/tfrecords/

!lm encode \
    --encoder gpt2 \
    --name  \
    ${DATASET_OUTPUT}/\* \
    ${TFRECORD_OUTPUT} 
```

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
