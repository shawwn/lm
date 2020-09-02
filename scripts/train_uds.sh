#!/usr/bin/env bash 

# clean utf8
CLEANTXT_INPUT=data/uds
CLEANTXT_OUTPUT=/tmp/cleantxt
lm cleantxt ${CLEANTXT_INPUT} ${CLEANTXT_OUTPUT} --force

# train encoder
TOKENIZER_INPUT=${CLEANTXT_OUTPUT}
TOKENIZER_OUTPUT=/tmp/tokenizer/
lm tokenize gpt2 ${CLEANTXT_OUTPUT}/

# converts to tfrecord 
ENCODE_INPUT=${CLEANTXT_OUTPUT}
ENCODE_OUTPUT=/tmp/tfrecord
lm encode --encoder gpt2 ${ENCODE_INPUT}/\*.\* ${ENCODE_OUTPUT}
