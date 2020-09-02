import os
from typing import Dict

import tensorflow as tf
from pydantic import AnyUrl
from pydantic.dataclasses import dataclass

from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast


@dataclass
class EncoderConfig:
    is_pretrained: bool
    location: AnyUrl


def fetch_encoder(config: EncoderConfig):
    if config.is_pretrained:
        return GPT2TokenizerFast.from_pretrained(config.location)

    return Tokenizer.from_file(config.location)


# GPT2Tokenizer and Tokenizer has different ways of fetching token ids
def encode(encoder, text):
    result = encoder.encode(text)
    if isinstance(result, list):
        return result
    return result.ids


def load_tokenizer(location):
    if tf.io.gfile.exists(os.path.join(location, "merges.txt")):
        # use tf gfile in case the dictionary is remote
        fastok = GPT2TokenizerFast.from_pretrained(location)
        fastok.add_special_tokens(
            {"eos_token": "[EOS]", "pad_token": "[PAD]", "unk_token": "[UNK]"}
        )
    else:
        if location.startswith("/"):
            raise ValueError("invalid location %s", location)
        else:
            fastok = GPT2TokenizerFast.from_pretrained(location)
    return fastok


def from_config(config: Dict):
    if config["kind"] == "hf":
        return load_tokenizer(config["location"])
    raise ValueError("invalid configuration")
